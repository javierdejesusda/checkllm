from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

IMAGE_RELEVANCE_SYSTEM_PROMPT = """You are an expert multimodal evaluator. Your job is to assess whether an image is relevant to the query or topic at hand.

Score from 0.0 to 1.0:
- 1.0 = The image is highly relevant; it directly depicts, illustrates, or addresses the query/topic
- 0.7 = The image is mostly relevant; it is related to the topic with minor tangential elements
- 0.5 = The image is partially relevant; it has some connection to the topic but significant portions are off-topic
- 0.3 = The image has low relevance; the connection to the query/topic is weak or incidental
- 0.0 = The image is completely irrelevant to the query or topic

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageRelevanceMetric:
    """Evaluates whether an image is relevant to the query or topic."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_RELEVANCE_SYSTEM_PROMPT

    async def evaluate(self, image_description: str, query: str) -> CheckResult:
        """Evaluate whether an image is relevant to a given query or topic.

        Args:
            image_description: A text description of the image content.
            query: The query or topic the image should be relevant to.

        Returns:
            A CheckResult with relevance score and reasoning.
        """
        prompt = (
            f"Query/Topic:\n{query}\n\n"
            f"Image Description:\n{image_description}\n\n"
            "Is the image relevant to the query or topic? Score it."
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
            metric_name="image_relevance",
        )
