from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

MULTIMODAL_FAITHFULNESS_SYSTEM_PROMPT = """You are an expert multimodal faithfulness evaluator. Your job is to assess whether a text output accurately and faithfully describes or references the content of an image, grounded in the source context.

Score from 0.0 to 1.0:
- 1.0 = The text output is completely faithful; every claim about the image is accurate and supported by what the image actually shows, consistent with the source context
- 0.7 = The text is mostly faithful with minor inaccuracies or slight overstatements about the image
- 0.5 = The text is partially faithful; some claims about the image are accurate but others are inaccurate or unsupported
- 0.3 = The text has low faithfulness; it makes several claims about the image that are not supported or are contradicted by the image
- 0.0 = The text is entirely unfaithful to the image; it misrepresents or fabricates the image content

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class MultimodalFaithfulnessMetric:
    """Evaluates cross-modal faithfulness between text output and image content."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = MULTIMODAL_FAITHFULNESS_SYSTEM_PROMPT

    async def evaluate(
        self,
        image_description: str,
        text_output: str,
        source_context: str,
    ) -> CheckResult:
        """Evaluate whether text accurately and faithfully describes image content.

        Args:
            image_description: A text description of the actual image content.
            text_output: The text output to evaluate for faithfulness to the image.
            source_context: The broader source context in which the image appears.

        Returns:
            A CheckResult with faithfulness score and reasoning.
        """
        prompt = (
            f"Source Context:\n{source_context}\n\n"
            f"Image Description (ground truth):\n{image_description}\n\n"
            f"Text Output to Evaluate:\n{text_output}\n\n"
            "Does the text output faithfully and accurately describe the image content? Score it."
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
            metric_name="multimodal_faithfulness",
        )
