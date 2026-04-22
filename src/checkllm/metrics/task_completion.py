from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

TASK_COMPLETION_SYSTEM_PROMPT = """You are an expert task completion evaluator. Your job is to assess whether an LLM's output actually accomplishes the user's stated goal. This is different from instruction-following (which focuses on format compliance) — you are evaluating whether the substantive goal was achieved.

Your evaluation process:
1. Identify the core goal or objective from the task description.
2. If constraints are provided, note each one as a requirement for full completion.
3. Evaluate whether the output achieves the primary goal, not just whether it follows formatting rules.
4. Check each constraint to see if it was satisfied in substance.
5. Consider partial completion: did the output make meaningful progress toward the goal even if it didn't fully achieve it?

Score from 0.0 to 1.0:
- 1.0 = The task is fully completed. The primary goal is achieved and all constraints are satisfied. The user would consider this done.
- 0.8 = The task is substantially completed. The primary goal is achieved with minor omissions or one constraint not fully met.
- 0.5 = The task is partially completed. The output addresses the goal but leaves significant work undone or misses important constraints.
- 0.3 = The task is barely started. The output shows some understanding of the goal but fails to make meaningful progress.
- 0.0 = The task is not completed at all. The output is off-topic, empty, or completely fails to address the stated goal.

Key evaluation criteria:
1. Primary goal achievement: Is the core objective met?
2. Constraint satisfaction: Are all specified constraints respected?
3. Completeness: Would the user need to do significant additional work?
4. Correctness: Is the completed work accurate and usable?
5. Distinguish from format compliance: A beautifully formatted but wrong answer should score low.

Respond with JSON: {"score": <float>, "reasoning": "<explanation of goal achievement and constraint satisfaction>"}"""


class TaskCompletionMetric:
    """Evaluates whether the LLM accomplished the user's stated goal."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TASK_COMPLETION_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        task_description: str,
        constraints: list[str] | None = None,
    ) -> CheckResult:
        constraints_section = ""
        if constraints:
            formatted_constraints = "\n".join(f"  {i}. {c}" for i, c in enumerate(constraints, 1))
            constraints_section = f"\nConstraints:\n{formatted_constraints}\n"

        prompt = (
            f"Task Description:\n{task_description}\n"
            f"{constraints_section}\n"
            f"Output to evaluate:\n{output}\n\n"
            "Did the output accomplish the stated task goal? "
            "Evaluate goal achievement and constraint satisfaction. Score it."
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
            metric_name="task_completion",
        )
