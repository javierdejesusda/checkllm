"""Image captioning quality metric.

Scores a generated caption against the actual image (and optionally a
reference caption). Uses a vision-capable LLM judge.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

IMAGE_CAPTIONING_QUALITY_SYSTEM_PROMPT = """You are an expert evaluator of image captioning. Judge the quality of a generated caption relative to the image (and optional reference caption).

Score from 0.0 to 1.0:
- 1.0 = Caption is accurate, salient, specific, and fluent; captures the main subject(s), setting, and notable details.
- 0.7 = Accurate and relevant but misses some salient detail, is slightly generic, or is overly verbose.
- 0.5 = Partially correct; mentions the main subject but omits important context or includes minor errors.
- 0.3 = Mostly incorrect or vague; mentions only loosely related content.
- 0.0 = Wrong, unrelated, or fabricated.

Evaluate: accuracy, salience (does it describe the most important content?), specificity, fluency. Penalize hallucinated objects/actions. If a reference caption is given, use it as a quality guide but do NOT require exact wording.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageCaptioningQualityMetric:
    """Scores caption quality against the image using a vision judge."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_CAPTIONING_QUALITY_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        caption: str,
        reference_caption: str | None = None,
    ) -> CheckResult:
        """Evaluate a caption against the image content.

        Args:
            image: Image source(s).
            caption: The generated caption to evaluate.
            reference_caption: Optional ground-truth caption used as a quality
                reference (the model is told not to require exact wording).

        Returns:
            A ``CheckResult`` with the caption quality score.
        """
        payloads = _ensure_payloads(image)
        prompt_parts = [f"Generated caption:\n{caption}"]
        if reference_caption:
            prompt_parts.append(f"Reference caption:\n{reference_caption}")
        prompt_parts.append("Score the generated caption.")
        prompt = "\n\n".join(prompt_parts)

        start = time.perf_counter_ns()
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
            metric_name="image_captioning_quality",
        )
