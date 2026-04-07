from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SYSTEM_PROMPT = """You are an expert evaluator of tool call correctness. Assess whether the actual tool calls match the expected calls in terms of tool names, argument names, and argument values.

Score from 0.0 to 1.0:
- 1.0 = Tool names, argument names, and argument values all match the expected calls exactly
- 0.5 = Tool names are correct but some argument names or values are wrong or missing
- 0.0 = Wrong tool names called, or argument names and values are significantly incorrect

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ArgumentCorrectnessMetric:
    """Evaluates whether tool calls match expected calls in name and arguments."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SYSTEM_PROMPT

    async def evaluate(self, tool_calls: str, expected_calls: str) -> CheckResult:
        """Evaluate argument correctness of tool calls.

        Args:
            tool_calls: The actual tool calls made.
            expected_calls: The expected tool calls.

        Returns:
            A CheckResult with the evaluation outcome.
        """
        prompt = (
            f"Expected tool calls:\n{expected_calls}\n\n"
            f"Actual tool calls:\n{tool_calls}\n\n"
            "Are the tool names, argument names, and argument values correct? Score it."
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
            metric_name="argument_correctness",
        )
