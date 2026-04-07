from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SYSTEM_PROMPT = """You are an expert evaluator of agent plans. Assess whether the plan is logical, complete, well-structured, and addresses the given task.

Score from 0.0 to 1.0:
- 1.0 = Plan is fully logical, complete, well-structured, and directly addresses every aspect of the task
- 0.5 = Plan addresses the task but has gaps, unclear steps, or minor logical issues
- 0.0 = Plan is illogical, incomplete, poorly structured, or fails to address the task

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class PlanQualityMetric:
    """Evaluates the quality of an agent plan against a given task."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SYSTEM_PROMPT

    async def evaluate(self, plan: str, task: str) -> CheckResult:
        """Evaluate plan quality.

        Args:
            plan: The plan to evaluate.
            task: The task the plan is meant to address.

        Returns:
            A CheckResult with the evaluation outcome.
        """
        prompt = (
            f"Task:\n{task}\n\n"
            f"Plan:\n{plan}\n\n"
            "Evaluate whether this plan is logical, complete, well-structured, and addresses the task. Score it."
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
            metric_name="plan_quality",
        )
