from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CORRECTNESS_SYSTEM_PROMPT = """You are an expert answer correctness evaluator. Your job is to assess whether an LLM output is correct compared to the expected/reference answer.

Score from 0.0 to 1.0:
- 1.0 = Output conveys the same essential information as the expected answer, even if phrased differently
- 0.5 = Output partially matches the expected answer but misses key information or adds incorrect information
- 0.0 = Output is completely incorrect or contradicts the expected answer

Focus on semantic correctness, not exact wording. Different phrasing of the same facts should score high.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class CorrectnessMetric:
    """Checks whether LLM output is correct compared to an expected/reference answer."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CORRECTNESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, expected: str) -> CheckResult:
        prompt = (
            f"Expected/Reference answer:\n{expected}\n\n"
            f"Actual output to evaluate:\n{output}\n\n"
            "Is the output correct compared to the expected answer? Score it."
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
            metric_name="correctness",
        )
