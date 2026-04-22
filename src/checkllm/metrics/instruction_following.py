from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

INSTRUCTION_FOLLOWING_SYSTEM_PROMPT = """You are an expert instruction-following evaluator. Your job is to assess whether the output faithfully follows the given instructions, including format requirements, constraints, style guidelines, and any other explicit directives.

Score from 0.0 to 1.0:
- 1.0 = The output perfectly follows every instruction. All format requirements, constraints, style guidelines, and directives are satisfied.
- 0.8 = The output follows most instructions with only minor deviations that don't materially affect the result (e.g., slight formatting differences).
- 0.5 = The output follows some instructions but misses or violates others. Key constraints may be partially met.
- 0.3 = The output largely ignores the instructions. Only superficial compliance is observed.
- 0.0 = The output completely disregards the instructions or does the opposite of what was asked.

Key evaluation criteria:
1. Format compliance: Does the output match the requested format (e.g., JSON, bullet points, numbered list, specific structure)?
2. Constraint adherence: Are length limits, word counts, or other quantitative constraints respected?
3. Style compliance: Does the output match the requested tone, formality, or writing style?
4. Content directives: Are all requested topics, sections, or elements included?
5. Negative constraints: Does the output avoid anything that was explicitly prohibited?
6. Ordering and structure: Is the information organized as instructed?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class InstructionFollowingMetric:
    """Checks whether the output follows given instructions (format, constraints, style)."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = INSTRUCTION_FOLLOWING_SYSTEM_PROMPT

    async def evaluate(self, output: str, instructions: str) -> CheckResult:
        prompt = (
            f"Instructions:\n{instructions}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            "Does the output follow the given instructions? "
            "Check format, constraints, style, and content requirements. Score it."
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
            metric_name="instruction_following",
        )
