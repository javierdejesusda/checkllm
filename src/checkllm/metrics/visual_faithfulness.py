"""Visual faithfulness metric.

Parallel of the text ``faithfulness`` metric where the grounding modality is
the image(s) instead of text context. Given an image and an output, scores
whether the output's claims are supported by the image.

This is distinct from the existing ``multimodal_faithfulness`` metric, which
operates on a text description of the image; ``visual_faithfulness`` sends
the actual image to a vision-capable judge.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

VISUAL_FAITHFULNESS_SYSTEM_PROMPT = """You are an expert evaluator of visual faithfulness. You will be shown one or more images and a text output (typically an answer to a user query that references the image). Decide whether the output is faithful to the image(s).

Score from 0.0 to 1.0:
- 1.0 = Every claim the output makes about the image is directly supported by the image.
- 0.7 = Mostly faithful with minor unsupported embellishments.
- 0.5 = Mixed: some supported claims, some unsupported.
- 0.2 = Mostly unfaithful; claims contradict or fabricate the image.
- 0.0 = Entirely unfaithful.

Treat a concise correct answer (a number, short name, yes/no) as 1.0 when it matches what the image shows.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class VisualFaithfulnessMetric:
    """Scores faithfulness of text output to grounded image(s)."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = VISUAL_FAITHFULNESS_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        output: str,
        query: str | None = None,
    ) -> CheckResult:
        """Evaluate faithfulness of ``output`` to the image(s).

        Args:
            image: Image source(s) used as the grounding modality.
            output: The text output to evaluate.
            query: Optional query the output is answering.

        Returns:
            A ``CheckResult`` with the faithfulness score.
        """
        payloads = _ensure_payloads(image)
        parts = []
        if query:
            parts.append(f"Query:\n{query}")
        parts.append(f"Output to evaluate:\n{output}")
        parts.append("Is the output faithful to the image(s)? Score it.")
        prompt = "\n\n".join(parts)

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
            metric_name="visual_faithfulness",
        )
