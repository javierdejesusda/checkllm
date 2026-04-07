from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SYSTEM_PROMPT = """You are an expert evaluator of agent execution efficiency. Assess whether the steps taken to complete a task were efficient and free of unnecessary redundancy.

Score from 0.0 to 1.0:
- 1.0 = All steps were necessary and sufficient; no redundant or wasteful steps
- 0.5 = Some redundant or unnecessary steps present but task was still completed reasonably
- 0.0 = Many redundant, unnecessary, or inefficient steps that could have been avoided

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class StepEfficiencyMetric:
    """Evaluates whether steps taken were efficient relative to the task."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SYSTEM_PROMPT

    async def evaluate(self, steps: list[str], task: str) -> CheckResult:
        """Evaluate step efficiency.

        Args:
            steps: The list of steps taken.
            task: The task the steps were meant to complete.

        Returns:
            A CheckResult with the evaluation outcome.
        """
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = (
            f"Task:\n{task}\n\n"
            f"Steps taken:\n{numbered}\n\n"
            "Were these steps efficient? Were any redundant or unnecessary? Score it."
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
            metric_name="step_efficiency",
        )
