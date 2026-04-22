from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

IMAGE_REFERENCE_SYSTEM_PROMPT = """You are an expert evaluator for image generation quality. Your job is to assess how well a generated image matches a reference image, based on text descriptions of both images.

Score from 0.0 to 1.0:
- 1.0 = The generated image perfectly matches the reference. All key elements, composition, colors, and style align.
- 0.8 = The generated image closely matches the reference with minor differences.
- 0.5 = The generated image partially matches the reference. Some elements align but others diverge.
- 0.3 = The generated image has little resemblance to the reference.
- 0.0 = The generated image does not match the reference at all.

Key evaluation criteria:
1. Do the main subjects match between generated and reference images?
2. Is the composition similar (layout, positioning, proportions)?
3. Do colors, lighting, and style align?
4. Are important details preserved?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageReferenceMetric:
    """Evaluates how well a generated image matches a reference image.

    Compares descriptions of the generated and reference images to assess
    alignment in subject, composition, colors, and style.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_REFERENCE_SYSTEM_PROMPT

    async def evaluate(
        self,
        generated_image_desc: str,
        reference_image_desc: str,
    ) -> CheckResult:
        """Evaluate how well a generated image matches a reference image.

        Args:
            generated_image_desc: A text description of the generated image.
            reference_image_desc: A text description of the reference image.

        Returns:
            A CheckResult with reference match score and reasoning.
        """
        prompt = (
            f"Reference Image Description:\n{reference_image_desc}\n\n"
            f"Generated Image Description:\n{generated_image_desc}\n\n"
            "How well does the generated image match the reference image? "
            "Compare subjects, composition, colors, style, and details. Score it."
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
            metric_name="image_reference",
        )
