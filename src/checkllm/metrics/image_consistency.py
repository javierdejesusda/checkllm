"""Image consistency metric.

For multi-image inputs, scores whether a single response is consistent with
ALL supplied images — not only one. Useful for comparative reasoning, image
sets, and multi-frame workflows.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

IMAGE_CONSISTENCY_SYSTEM_PROMPT = """You are an expert multi-image evaluator. You will be shown two or more images and a text response. Judge whether the response is consistent with ALL of the images, not just one.

Score from 0.0 to 1.0:
- 1.0 = The response is consistent with every image. Claims about "the images" are true across the set; claims about individual images are correctly attributed.
- 0.7 = Consistent with most images but contains at least one claim that conflicts with one of them.
- 0.5 = Consistent with some images but wrong for others; or correct overall but mis-attributes details.
- 0.2 = Only matches one image; ignores or contradicts the rest.
- 0.0 = Inconsistent with all of the images.

Pay particular attention to comparative claims ("all", "both", "the first", "the second", "none of them"): verify each against the full set.

Respond with JSON: {"score": <float>, "reasoning": "<explanation, calling out any inconsistencies>"}"""


class ImageConsistencyMetric:
    """Scores whether a response is consistent with every image in a set."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_CONSISTENCY_SYSTEM_PROMPT

    async def evaluate(
        self,
        images: Iterable[ImageSource],
        response: str,
        query: str | None = None,
    ) -> CheckResult:
        """Evaluate cross-image consistency of ``response``.

        Args:
            images: Two or more image sources.
            response: The model response to evaluate.
            query: Optional query the response is answering.

        Returns:
            A ``CheckResult`` with the consistency score.

        Raises:
            ValueError: If fewer than two images are supplied.
        """
        payloads = _ensure_payloads(images)
        if len(payloads) < 2:
            raise ValueError("image_consistency requires at least two images")

        parts = []
        if query:
            parts.append(f"Query:\n{query}")
        parts.append(f"Response to evaluate:\n{response}")
        parts.append(
            f"There are {len(payloads)} images. "
            "Score how consistent the response is with ALL of them."
        )
        prompt = "\n\n".join(parts)

        start = time.perf_counter_ns()
        judge_response = await call_vision_judge(
            self.judge, prompt=prompt, images=payloads, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=judge_response.score >= self.threshold,
            score=judge_response.score,
            reasoning=judge_response.reasoning,
            cost=judge_response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="image_consistency",
        )
