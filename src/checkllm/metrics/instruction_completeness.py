from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

INSTRUCTION_COMPLETENESS_SYSTEM_PROMPT = """You are an expert instruction compliance evaluator. Your task is to assess whether a response follows ALL instructions from a given list of instructions, not just some of them.

This is different from general instruction following, which evaluates a single instruction string. Instruction completeness specifically handles lists of discrete, enumerable instructions and checks each one individually for compliance.

Evaluation process:
1. Review each instruction in the provided list.
2. For each instruction, determine if the response:
   - FOLLOWED: The instruction is clearly and fully satisfied.
   - PARTIALLY FOLLOWED: The instruction is acknowledged but not fully satisfied.
   - IGNORED: The instruction is not addressed at all.
   - VIOLATED: The response does the opposite of what was instructed.
3. Compute a score based on the proportion of fully followed instructions.

Score from 0.0 to 1.0:
- 1.0 = Every instruction is fully followed.
- 0.8 = Most instructions are followed; one may be only partially addressed.
- 0.5 = About half the instructions are followed.
- 0.3 = Most instructions are ignored or only partially followed.
- 0.0 = No instructions are followed, or instructions are actively violated.

Key evaluation criteria:
1. Is each instruction independently verifiable in the response?
2. Are instructions followed in spirit, not just letter?
3. Are there conflicting instructions that make full compliance impossible?
4. Does partial compliance count, or must each instruction be fully met?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class InstructionCompletenessMetric:
    """Evaluates whether ALL instructions in a list were followed.

    Given a list of discrete instructions, checks each one individually
    for compliance and computes a score based on the proportion fully
    followed. Different from instruction-following which handles a single
    instruction string.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = INSTRUCTION_COMPLETENESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, instructions: list[str]) -> CheckResult:
        """Evaluate whether the output follows all listed instructions.

        Args:
            output: The response to evaluate.
            instructions: List of instruction strings to check for compliance.

        Returns:
            CheckResult with instruction completeness score.
        """
        instructions_str = "\n".join(f"{i + 1}. {inst}" for i, inst in enumerate(instructions))
        prompt = (
            f"Instructions to follow:\n{instructions_str}\n\n"
            f"Response to evaluate:\n{output}\n\n"
            "Check each instruction individually. Was each one followed, "
            "partially followed, or ignored? Score the overall completeness."
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
            metric_name="instruction_completeness",
        )
