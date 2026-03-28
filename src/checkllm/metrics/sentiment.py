from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SENTIMENT_SYSTEM_PROMPT = """You are an expert sentiment evaluator. Your job is to assess the sentiment of an LLM output on a scale where:

Score from 0.0 to 1.0:
- 1.0 = Very positive, enthusiastic, encouraging, optimistic
- 0.5 = Neutral, balanced, factual, neither positive nor negative
- 0.0 = Very negative, critical, pessimistic, discouraging

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class SentimentMetric:
    """Assesses the sentiment of LLM output on a 0-1 scale (0=negative, 1=positive)."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.5) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SENTIMENT_SYSTEM_PROMPT

    async def evaluate(self, output: str) -> CheckResult:
        prompt = (
            f"Output to evaluate:\n{output}\n\n"
            "What is the sentiment of this output? Score it."
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
            metric_name="sentiment",
        )
