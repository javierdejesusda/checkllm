from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

FACTUAL_CORRECTNESS_SYSTEM_PROMPT = """You are an expert factual correctness evaluator. Your task is to decompose both the response and the reference into individual atomic claims, then assess how many claims in the response are factually correct according to the reference.

This is different from faithfulness (which checks grounding against retrieved context). Factual correctness compares the response directly against a known-correct reference answer to verify claim-level accuracy.

Evaluation process:
1. Decompose the reference into atomic factual claims (individual, verifiable statements).
2. Decompose the response into atomic factual claims.
3. For each claim in the response, determine if it is:
   - CORRECT: Supported by the reference claims.
   - INCORRECT: Contradicted by the reference claims.
   - UNVERIFIABLE: Neither supported nor contradicted by the reference (novel claim).
4. Compute a score based on the proportion of correct claims.

Score from 0.0 to 1.0:
- 1.0 = Every atomic claim in the response is factually correct per the reference.
- 0.8 = Most claims are correct with minor inaccuracies that do not change the overall meaning.
- 0.5 = A mix of correct and incorrect claims.
- 0.3 = More incorrect or unverifiable claims than correct ones.
- 0.0 = The response is entirely incorrect or contradicts the reference.

Key evaluation criteria:
1. Are numerical values, dates, and proper nouns accurate?
2. Are causal relationships and logical connections correctly stated?
3. Are there fabricated details not present in the reference?
4. Is the overall factual narrative consistent with the reference?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class FactualCorrectnessMetric:
    """Evaluates claim-level factual correctness of a response against a reference.

    Decomposes both the response and a known-correct reference into atomic
    claims, then judges the accuracy of each claim in the response. This
    differs from faithfulness, which checks grounding against retrieved context.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = FACTUAL_CORRECTNESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, reference: str) -> CheckResult:
        """Evaluate factual correctness of output against a reference answer.

        Args:
            output: The response to evaluate.
            reference: The known-correct reference answer.

        Returns:
            CheckResult with claim-level accuracy score.
        """
        prompt = (
            f"Reference Answer:\n{reference}\n\n"
            f"Response to evaluate:\n{output}\n\n"
            "Decompose both into atomic claims and assess the factual "
            "correctness of each claim in the response. Score it."
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
            metric_name="factual_correctness",
        )
