"""OCR accuracy metric.

Scores how faithfully an extracted text block matches what is actually
written in an image. When a ground-truth transcript is provided, the metric
can also return a deterministic character-error-rate-based score without
calling a judge.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

OCR_ACCURACY_SYSTEM_PROMPT = """You are an expert OCR quality evaluator. You will be shown an image containing text and a candidate transcript of that text. Compare the candidate to the text actually visible in the image.

Score from 0.0 to 1.0:
- 1.0 = Perfect transcription: every visible character, punctuation mark, and line break is captured; no hallucinated text.
- 0.8 = Minor errors (occasional typos, swapped similar characters, minor punctuation drift).
- 0.5 = Significant errors; large portions are wrong, missing, or reordered.
- 0.2 = Mostly incorrect; only fragments match.
- 0.0 = Transcript is unrelated or fabricated.

Penalize hallucinated text that is not in the image. Ignore purely cosmetic whitespace differences unless they change meaning.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


def _character_similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity based on Levenshtein-normalized distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    len_a, len_b = len(a), len(b)
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)
    for i in range(1, len_a + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    distance = prev[len_b]
    denom = max(len_a, len_b)
    return max(0.0, 1.0 - distance / denom)


class OCRAccuracyMetric:
    """Scores OCR transcript faithfulness against the source image."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.85) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = OCR_ACCURACY_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource] | None,
        extracted_text: str,
        ground_truth: str | None = None,
    ) -> CheckResult:
        """Evaluate OCR transcript accuracy.

        Args:
            image: Image source(s). May be ``None`` when ``ground_truth`` is
                supplied for a pure string-based comparison.
            extracted_text: The OCR output to evaluate.
            ground_truth: Optional ground-truth transcript. When provided and
                no image is given, the metric returns a deterministic score
                based on character-error-rate; when both are provided the
                judge still drives the score and ``ground_truth`` is surfaced
                in the prompt as a hint.

        Returns:
            A ``CheckResult`` with the OCR accuracy score.
        """
        start = time.perf_counter_ns()

        if image is None:
            if ground_truth is None:
                raise ValueError(
                    "ocr_accuracy requires either an image or a ground_truth transcript"
                )
            score = _character_similarity(extracted_text.strip(), ground_truth.strip())
            elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
            return CheckResult(
                passed=score >= self.threshold,
                score=score,
                reasoning=(f"Character similarity vs ground truth: {score:.2f}"),
                cost=0.0,
                latency_ms=int(elapsed_ms),
                metric_name="ocr_accuracy",
            )

        payloads = _ensure_payloads(image)
        prompt_parts = [f"Candidate transcript:\n{extracted_text}"]
        if ground_truth:
            prompt_parts.append(f"Reference transcript (hint):\n{ground_truth}")
        prompt_parts.append("Score OCR faithfulness to the image.")
        prompt = "\n\n".join(prompt_parts)

        response = await call_vision_judge(
            self.judge, prompt=prompt, images=payloads, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="ocr_accuracy",
        )
