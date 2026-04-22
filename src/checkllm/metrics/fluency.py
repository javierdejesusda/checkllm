from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

FLUENCY_SYSTEM_PROMPT = """You are an expert language quality evaluator. Your job is to assess the fluency of an LLM output.

Score from 0.0 to 1.0:
- 1.0 = Output is perfectly fluent, natural, and reads like it was written by a skilled native speaker
- 0.5 = Output is understandable but has awkward phrasing, grammatical issues, or unnatural word choices
- 0.0 = Output is incoherent, ungrammatical, or very difficult to understand

Evaluate: grammar, word choice, sentence structure, naturalness.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class FluencyMetric:
    """Checks whether LLM output is fluent and well-written."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = FLUENCY_SYSTEM_PROMPT

    async def evaluate(self, output: str) -> CheckResult:
        prompt = (
            f"Output to evaluate:\n{output}\n\nIs this output fluent and well-written? Score it."
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
            metric_name="fluency",
        )
