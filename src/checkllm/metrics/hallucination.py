from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

HALLUCINATION_SYSTEM_PROMPT = """You are an expert hallucination evaluator. Your job is to assess whether an LLM output is grounded in the provided context.

Score from 0.0 to 1.0:
- 1.0 = Every claim in the output is directly supported by the context
- 0.5 = Some claims are supported, others are not
- 0.0 = The output contains claims completely unsupported by or contradicting the context

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class HallucinationMetric:
    """Checks whether LLM output is grounded in the provided context."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold

    async def evaluate(self, output: str, context: str) -> CheckResult:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            "Is the output grounded in the context? Score it."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=HALLUCINATION_SYSTEM_PROMPT
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="hallucination",
        )
