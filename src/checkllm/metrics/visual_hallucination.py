"""Visual hallucination detection metric.

Flags claims in a text response that are not grounded in the supplied
image(s). Returns a score where 1.0 means fully grounded and 0.0 means every
claim is fabricated or contradicted by the image.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

VISUAL_HALLUCINATION_SYSTEM_PROMPT = """You are an expert detector of visual hallucinations. You will be shown one or more images and a text response. Identify claims in the response that are NOT supported by the images — objects, attributes, counts, spatial relationships, colors, text, actions, or any other detail that cannot be verified from the images.

Return a groundedness score from 0.0 to 1.0:
- 1.0 = Every claim is directly supported by the image(s). No hallucination.
- 0.8 = Minor embellishments but the core content is grounded.
- 0.5 = Roughly half the claims are unsupported or drift from the image.
- 0.2 = Mostly hallucinated; only a few claims match the image.
- 0.0 = The response is entirely fabricated or contradicts the image.

Be strict — claims that "could be true" but are not actually visible should be treated as hallucinations.

Respond with JSON: {"score": <float>, "reasoning": "<explanation listing the hallucinated claims>"}"""


class VisualHallucinationMetric:
    """Scores visual groundedness; penalizes unsupported claims about images."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = VISUAL_HALLUCINATION_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        response: str,
        query: str | None = None,
    ) -> CheckResult:
        """Evaluate whether ``response`` is grounded in the image(s).

        Args:
            image: Image source(s) to ground against.
            response: The model response to check.
            query: Optional user query that elicited the response.

        Returns:
            A ``CheckResult`` where higher scores mean less hallucination.
        """
        payloads = _ensure_payloads(image)
        prompt_parts = []
        if query:
            prompt_parts.append(f"User query:\n{query}")
        prompt_parts.append(f"Response to evaluate:\n{response}")
        prompt_parts.append(
            "Identify any claims that are not supported by the image(s) and score groundedness."
        )
        prompt = "\n\n".join(prompt_parts)

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
            metric_name="visual_hallucination",
        )
