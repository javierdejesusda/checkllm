from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

GROUNDEDNESS_SYSTEM_PROMPT = """You are an expert groundedness evaluator. Your job is to assess how well each claim in the output is grounded across multiple source documents. Unlike simple hallucination detection, you should evaluate each distinct claim individually and aggregate the results.

Score from 0.0 to 1.0:
- 1.0 = Every claim in the output is fully grounded in at least one of the provided source documents. All facts, figures, and assertions can be traced back to the sources.
- 0.8 = The vast majority of claims are grounded. Only trivial or inconsequential statements lack direct source support (e.g., common knowledge transitions).
- 0.5 = Roughly half the claims are grounded. The output mixes well-sourced claims with unsupported ones.
- 0.3 = Few claims are grounded. The output mostly contains information not traceable to the sources.
- 0.0 = No claims are grounded. The output is entirely unsupported by any of the provided sources.

Evaluation approach (claim-by-claim):
1. Identify each distinct factual claim in the output.
2. For each claim, check whether ANY of the provided sources support it.
3. A claim is grounded if it is directly stated in or logically derivable from at least one source.
4. Consider cross-referencing between sources: a claim might be supported by combining information from multiple sources.
5. Note which claims are ungrounded and explain why.
6. The final score should reflect the proportion of grounded claims weighted by their importance.

Key evaluation criteria:
1. Claim identification: Break the output into individual factual claims.
2. Source matching: For each claim, identify the supporting source(s) or note its absence.
3. Accuracy of references: Are the claims accurately representing what the sources say?
4. Coverage: What proportion of claims are grounded?
5. Severity: How important are the ungrounded claims?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class GroundednessMetric:
    """Scores claim-by-claim grounding across multiple source documents.

    More nuanced than hallucination detection: evaluates each claim individually
    and supports multiple source documents for cross-referencing.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = GROUNDEDNESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, sources: list[str]) -> CheckResult:
        numbered_sources = "\n\n".join(
            f"Source {i + 1}:\n{text}" for i, text in enumerate(sources)
        )
        prompt = (
            f"Source Documents:\n{numbered_sources}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            "Evaluate the groundedness of the output claim by claim. "
            "For each claim, check if any source supports it. Score it."
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
            metric_name="groundedness",
        )
