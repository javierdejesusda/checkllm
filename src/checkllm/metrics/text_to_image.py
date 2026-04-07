from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

TEXT_TO_IMAGE_SYSTEM_PROMPT = """You are an expert text-to-image evaluation specialist. Your job is to assess how well a generated image matches the original generation prompt.

Score from 0.0 to 1.0:
- 1.0 = The generated image perfectly matches the prompt; all specified subjects, attributes, styles, and relationships are accurately depicted
- 0.7 = The image mostly matches the prompt; most key elements are present but some details are missing or slightly off
- 0.5 = The image partially matches the prompt; the main subject is captured but important details, styles, or attributes are missing
- 0.3 = The image poorly matches the prompt; only superficial or coincidental elements of the prompt are reflected
- 0.0 = The image does not match the prompt at all; the content is unrelated or completely contradicts the prompt

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class TextToImageMetric:
    """Evaluates how well a generated image matches the generation prompt."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TEXT_TO_IMAGE_SYSTEM_PROMPT

    async def evaluate(
        self, image_description: str, original_prompt: str
    ) -> CheckResult:
        """Evaluate how well a generated image matches the original prompt.

        Args:
            image_description: A text description of the generated image content.
            original_prompt: The original text prompt used to generate the image.

        Returns:
            A CheckResult with prompt-adherence score and reasoning.
        """
        prompt = (
            f"Original Generation Prompt:\n{original_prompt}\n\n"
            f"Generated Image Description:\n{image_description}\n\n"
            "How well does the generated image match the original prompt? Score it."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="text_to_image",
        )
