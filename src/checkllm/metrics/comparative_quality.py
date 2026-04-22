from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

COMPARATIVE_QUALITY_SYSTEM_PROMPT = """You are an expert comparative evaluator. Your task is to compare two responses and judge which one is better according to specified evaluation criteria.

This is a pairwise comparison metric. Rather than scoring a single output in isolation, you compare two outputs head-to-head on the given criteria to determine which is superior.

Evaluation process:
1. Read both responses carefully.
2. For each criterion specified, compare the two responses.
3. Determine which response better satisfies each criterion.
4. Produce an overall score reflecting the relative quality.

Score interpretation (0.0 to 1.0):
- 1.0 = Response A is clearly and decisively better than Response B on all criteria.
- 0.8 = Response A is better on most criteria; B may be slightly better on one minor point.
- 0.5 = The responses are roughly equal in quality, or each is better on different criteria.
- 0.2 = Response B is better on most criteria.
- 0.0 = Response B is clearly and decisively better than Response A on all criteria.

Key evaluation criteria to apply:
1. Score based ONLY on the specified criteria, not general quality.
2. Be objective and avoid position bias (A appearing first should not matter).
3. Consider both substance and presentation when criteria are ambiguous.
4. If criteria conflict, weigh them equally unless otherwise specified.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ComparativeQualityMetric:
    """Compares two outputs and judges which is better on specified criteria.

    Performs pairwise comparison rather than absolute scoring. Returns a
    score where >0.5 means output A is better, <0.5 means output B is
    better, and 0.5 means they are roughly equal.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.5) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = COMPARATIVE_QUALITY_SYSTEM_PROMPT

    async def evaluate(self, output_a: str, output_b: str, criteria: str) -> CheckResult:
        """Compare two outputs on specified criteria.

        Args:
            output_a: The first response to compare.
            output_b: The second response to compare.
            criteria: The evaluation criteria for comparison.

        Returns:
            CheckResult where score >0.5 means A is better, <0.5 means B
            is better.
        """
        prompt = (
            f"Evaluation Criteria:\n{criteria}\n\n"
            f"Response A:\n{output_a}\n\n"
            f"Response B:\n{output_b}\n\n"
            "Compare these two responses on the specified criteria. "
            "Score >0.5 means A is better, <0.5 means B is better, "
            "0.5 means they are equal. Score it."
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
            metric_name="comparative_quality",
        )
