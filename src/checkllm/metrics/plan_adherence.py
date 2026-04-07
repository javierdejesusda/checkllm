from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SYSTEM_PROMPT = """You are an expert evaluator of plan execution. Assess whether the execution trace faithfully followed the given plan, without skipping or significantly reordering steps.

Score from 0.0 to 1.0:
- 1.0 = Execution followed the plan exactly; all steps were completed in the correct order
- 0.5 = Execution mostly followed the plan but skipped minor steps or reordered some steps
- 0.0 = Execution significantly deviated from the plan; major steps skipped or reordered

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class PlanAdherenceMetric:
    """Evaluates whether an execution trace adhered to the given plan."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SYSTEM_PROMPT

    async def evaluate(self, plan: str, execution_trace: str) -> CheckResult:
        """Evaluate plan adherence of an execution trace.

        Args:
            plan: The original plan that should have been followed.
            execution_trace: The actual execution trace to evaluate.

        Returns:
            A CheckResult with the evaluation outcome.
        """
        prompt = (
            f"Plan:\n{plan}\n\n"
            f"Execution trace:\n{execution_trace}\n\n"
            "Did the execution follow the plan? Were steps skipped or reordered? Score it."
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
            metric_name="plan_adherence",
        )
