from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

PROMPT_ALIGNMENT_SYSTEM_PROMPT = """You are an expert evaluator for instruction following. Your job is to assess whether the output follows a specific instruction from the system prompt.

Evaluate strictly:
- If the instruction is clearly followed, score 1.0.
- If the instruction is partially followed, score 0.5.
- If the instruction is not followed at all, score 0.0.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class PromptAlignmentMetric:
    """Evaluates whether an output follows the instructions in a system prompt.

    Checks each instruction individually against the output and computes
    the fraction of instructions that were followed.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = PROMPT_ALIGNMENT_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        system_prompt: str,
        instructions: list[str],
    ) -> CheckResult:
        """Evaluate whether the output follows the given instructions.

        Args:
            output: The LLM output to evaluate.
            system_prompt: The system prompt that was given to the LLM.
            instructions: A list of specific instructions to check compliance for.

        Returns:
            A CheckResult with alignment score and per-instruction details.
        """
        start = time.perf_counter_ns()
        instruction_scores: list[float] = []
        instruction_details: list[str] = []
        total_cost = 0.0

        for i, instruction in enumerate(instructions):
            prompt = (
                f"System Prompt:\n{system_prompt}\n\n"
                f"Instruction to Check:\n{instruction}\n\n"
                f"LLM Output:\n{output}\n\n"
                "Does the output follow this specific instruction? Score it."
            )
            response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
            instruction_scores.append(response.score)
            instruction_details.append(
                f"Instruction {i + 1} ({instruction[:50]}): "
                f"{response.score:.2f} - {response.reasoning}"
            )
            total_cost += response.cost

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        if instruction_scores:
            aggregate = sum(instruction_scores) / len(instruction_scores)
        else:
            aggregate = 0.0

        reasoning = (
            f"Instructions followed: "
            f"{sum(1 for s in instruction_scores if s >= 0.5)}/{len(instructions)}. "
            + " | ".join(instruction_details)
        )

        return CheckResult(
            passed=aggregate >= self.threshold,
            score=aggregate,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="prompt_alignment",
        )
