from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

IMAGE_EDITING_SYSTEM_PROMPT = """You are an expert evaluator for image editing quality. Your job is to assess whether an image edit correctly followed the editing instruction, based on descriptions of the original and edited images.

Score from 0.0 to 1.0:
- 1.0 = The edit perfectly follows the instruction. The edited image reflects exactly what was requested.
- 0.8 = The edit mostly follows the instruction with minor deviations.
- 0.5 = The edit partially follows the instruction. Some aspects are correct but others are wrong or missing.
- 0.3 = The edit poorly follows the instruction. Most of the requested changes are incorrect or absent.
- 0.0 = The edit does not follow the instruction at all, or the image was made worse.

Key evaluation criteria:
1. Does the edited image reflect the requested changes?
2. Were unrelated parts of the image preserved?
3. Is the quality of the edit acceptable?
4. Are there any unintended artifacts or changes?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageEditingMetric:
    """Evaluates quality of image editing based on an instruction.

    Compares the original and edited image descriptions against the editing
    instruction to determine if the edit was performed correctly.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_EDITING_SYSTEM_PROMPT

    async def evaluate(
        self,
        editing_instruction: str,
        original_image_desc: str,
        edited_image_desc: str,
    ) -> CheckResult:
        """Evaluate whether an image edit correctly followed the instruction.

        Args:
            editing_instruction: The instruction describing what edits to make.
            original_image_desc: A text description of the original image.
            edited_image_desc: A text description of the edited image.

        Returns:
            A CheckResult with editing quality score and reasoning.
        """
        prompt = (
            f"Editing Instruction:\n{editing_instruction}\n\n"
            f"Original Image Description:\n{original_image_desc}\n\n"
            f"Edited Image Description:\n{edited_image_desc}\n\n"
            "Did the edit correctly follow the instruction? Were the right changes "
            "made while preserving unrelated parts of the image? Score it."
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
            metric_name="image_editing",
        )
