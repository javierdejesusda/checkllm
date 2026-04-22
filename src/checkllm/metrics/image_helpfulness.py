from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

IMAGE_HELPFULNESS_SYSTEM_PROMPT = """You are an expert multimodal evaluator. Your job is to assess whether an image aids user comprehension in the context of the user's query.

Score from 0.0 to 1.0:
- 1.0 = The image is highly helpful; it directly clarifies, illustrates, or adds essential information to answer the user's query
- 0.7 = The image is mostly helpful; it provides useful context but may not be strictly necessary
- 0.5 = The image is somewhat helpful; it adds some value but its contribution to comprehension is limited
- 0.3 = The image provides minimal help; it adds little value toward answering the query
- 0.0 = The image is not helpful at all; it is irrelevant, misleading, or distracts from the query

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageHelpfulnessMetric:
    """Evaluates whether an image aids user comprehension given a query."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_HELPFULNESS_SYSTEM_PROMPT

    async def evaluate(self, image_description: str, query: str) -> CheckResult:
        """Evaluate whether an image is helpful for a given user query.

        Args:
            image_description: A text description of the image content.
            query: The user's query or question being addressed.

        Returns:
            A CheckResult with helpfulness score and reasoning.
        """
        prompt = (
            f"User Query:\n{query}\n\n"
            f"Image Description:\n{image_description}\n\n"
            "Does this image aid user comprehension in the context of the query? Score it."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="image_helpfulness",
        )
