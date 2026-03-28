from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

RELEVANCE_SYSTEM_PROMPT = """You are an expert relevance evaluator. Your job is to assess whether an LLM output is relevant to the user's query.

Score from 0.0 to 1.0:
- 1.0 = Output directly and completely answers the query
- 0.5 = Output partially addresses the query but misses key aspects
- 0.0 = Output is completely unrelated to the query

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class RelevanceMetric:
    """Checks whether LLM output is relevant to the query."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold

    async def evaluate(self, output: str, query: str) -> CheckResult:
        prompt = (
            f"Query:\n{query}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            "Is the output relevant to the query? Score it."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=RELEVANCE_SYSTEM_PROMPT
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="relevance",
        )
