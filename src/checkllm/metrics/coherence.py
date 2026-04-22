from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

COHERENCE_SYSTEM_PROMPT = """You are an expert coherence evaluator. Your job is to assess whether an LLM output is logically coherent and well-structured.

Score from 0.0 to 1.0:
- 1.0 = Output is perfectly coherent with a clear logical flow, consistent points, and well-organized structure
- 0.5 = Output is partially coherent but has some logical gaps, contradictions, or disorganized sections
- 0.0 = Output is incoherent, self-contradictory, or completely disorganized

Evaluate: logical flow, consistency, structure, absence of contradictions.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class CoherenceMetric:
    """Checks whether LLM output is logically coherent and well-structured."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = COHERENCE_SYSTEM_PROMPT

    async def evaluate(self, output: str) -> CheckResult:
        prompt = (
            f"Output to evaluate:\n{output}\n\n"
            "Is this output logically coherent and well-structured? Score it."
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
            metric_name="coherence",
        )
