from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SYSTEM_PROMPT = """You are an expert evaluator. Determine whether the given output fully achieves the stated goal.

Score from 0.0 to 1.0:
- 1.0 = Output completely and correctly achieves the stated goal
- 0.5 = Output partially achieves the goal but misses key elements
- 0.0 = Output fails to achieve the goal or produces an incorrect result

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class GoalAccuracyMetric:
    """Evaluates whether an output achieves a stated goal."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SYSTEM_PROMPT

    async def evaluate(self, output: str, goal: str) -> CheckResult:
        """Evaluate whether output achieves the goal.

        Args:
            output: The output to evaluate.
            goal: The goal the output should achieve.

        Returns:
            A CheckResult with the evaluation outcome.
        """
        prompt = (
            f"Goal:\n{goal}\n\n"
            f"Output:\n{output}\n\n"
            "Did the output achieve the stated goal? Score it."
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
            metric_name="goal_accuracy",
        )
