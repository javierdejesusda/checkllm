from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CITATION_ACCURACY_SYSTEM_PROMPT = """You are an expert citation accuracy evaluator. Your task is to verify that citations or references in a response correctly correspond to the provided source texts.

Many LLM outputs include citations (e.g., [1], [Source A], footnotes) that point to specific sources. This metric checks whether each citation actually references material found in the cited source, and whether the cited claim is accurately attributed.

Evaluation process:
1. Identify all citations or references in the response (numbered references, named sources, inline citations, etc.).
2. For each citation, locate the corresponding source text.
3. Verify that the cited claim is actually present in or supported by the referenced source.
4. Check for miscitations (citing the wrong source for a claim).
5. Check for fabricated citations (referencing sources that do not exist in the provided list).
6. Assess overall citation accuracy.

Score from 0.0 to 1.0:
- 1.0 = Every citation correctly references material in the cited source. No fabricated or miscited references.
- 0.8 = Most citations are accurate; one minor miscitation or imprecise reference.
- 0.5 = About half the citations are accurate; several point to wrong sources or misrepresent source content.
- 0.3 = Most citations are inaccurate or fabricated.
- 0.0 = All citations are fabricated or grossly incorrect.

Key evaluation criteria:
1. Does each citation point to a real, provided source?
2. Does the cited source actually contain the claimed information?
3. Is the cited information accurately represented (not taken out of context)?
4. Are there claims that should be cited but are not?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class CitationAccuracyMetric:
    """Verifies that citations in the output correctly reference provided sources.

    Checks each citation or reference in the response against the provided
    source texts to ensure claims are accurately attributed and no citations
    are fabricated or misattributed.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CITATION_ACCURACY_SYSTEM_PROMPT

    async def evaluate(self, output: str, sources: list[str]) -> CheckResult:
        """Evaluate citation accuracy against provided sources.

        Args:
            output: The response containing citations to verify.
            sources: List of source texts that citations should reference.

        Returns:
            CheckResult with citation accuracy score.
        """
        sources_str = "\n\n".join(f"[Source {i + 1}]:\n{src}" for i, src in enumerate(sources))
        prompt = (
            f"Source Texts:\n{sources_str}\n\n"
            f"Response with citations to verify:\n{output}\n\n"
            "Verify each citation in the response against the provided sources. "
            "Are citations accurate and properly attributed? Score it."
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
            metric_name="citation_accuracy",
        )
