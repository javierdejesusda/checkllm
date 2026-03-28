from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

RUBRIC_SYSTEM_PROMPT = """You are an expert evaluator. Your job is to assess whether an LLM output meets the specified quality criteria.

Score from 0.0 to 1.0:
- 1.0 = Output fully meets all specified criteria
- 0.5 = Output partially meets the criteria
- 0.0 = Output does not meet the criteria at all

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class RubricMetric:
    """Checks whether LLM output meets user-defined quality criteria."""

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    async def evaluate(
        self, output: str, criteria: str, threshold: float = 0.8
    ) -> CheckResult:
        prompt = (
            f"Criteria:\n{criteria}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            "Does the output meet the criteria? Score it."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=RUBRIC_SYSTEM_PROMPT
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="rubric",
        )
