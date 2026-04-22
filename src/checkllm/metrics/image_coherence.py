from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

IMAGE_COHERENCE_SYSTEM_PROMPT = """You are an expert multimodal evaluator. Your job is to assess whether the content described in an image is coherent with and appropriately supports the surrounding text context.

Score from 0.0 to 1.0:
- 1.0 = The image content is fully coherent with the text; it naturally fits the topic, tone, and message of the surrounding text
- 0.7 = The image is mostly coherent with minor inconsistencies or only loosely related elements
- 0.5 = The image is somewhat coherent but there are notable disconnects between the image and text
- 0.3 = The image has little coherence with the text; the connection is tenuous or misleading
- 0.0 = The image content is completely incoherent with or contradicts the surrounding text

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageCoherenceMetric:
    """Evaluates whether image content is coherent with surrounding text context."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_COHERENCE_SYSTEM_PROMPT

    async def evaluate(self, image_description: str, text_context: str) -> CheckResult:
        """Evaluate coherence between an image and its surrounding text.

        Args:
            image_description: A text description of the image content.
            text_context: The surrounding text context where the image appears.

        Returns:
            A CheckResult with coherence score and reasoning.
        """
        prompt = (
            f"Image Description:\n{image_description}\n\n"
            f"Surrounding Text Context:\n{text_context}\n\n"
            "Is the image coherent with the surrounding text? Score it."
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
            metric_name="image_coherence",
        )
