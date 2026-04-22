from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONTEXTUAL_PRECISION_SYSTEM_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems, specializing in contextual precision. Your job is to assess whether the most relevant pieces of retrieved context are ranked higher (appear earlier) in the context list.

Your evaluation process:
1. Read the user query and the expected (ground-truth) answer.
2. For each retrieved document (numbered in order), determine whether it is relevant to producing the expected answer.
3. A document is "relevant" if it contains information that directly supports or is needed to produce the expected answer.
4. Assess whether relevant documents appear earlier in the ranking and irrelevant documents appear later.
5. Compute a precision-weighted score: relevant documents appearing at the top contribute more to a high score than relevant documents buried at the bottom.

Score from 0.0 to 1.0:
- 1.0 = All relevant documents are ranked at the top, with no irrelevant documents preceding them. Perfect ranking.
- 0.8 = Most relevant documents are ranked highly, with only minor ranking imperfections.
- 0.5 = Relevant documents are scattered throughout the ranking with no clear prioritization. Mixed ordering.
- 0.3 = Most relevant documents are ranked below irrelevant ones. Poor retrieval ranking.
- 0.0 = All relevant documents are ranked at the bottom, or no relevant documents are present at all.

Key evaluation criteria:
1. Are the documents that directly answer the query ranked first?
2. Are irrelevant or tangential documents pushed to lower positions?
3. Would reordering the documents improve the ability to answer the query?
4. Consider the weighted precision: a relevant document at position 1 is far more valuable than one at position 10.

Respond with JSON: {"score": <float>, "reasoning": "<explanation of relevance assessment for each document and overall ranking quality>"}"""


class ContextualPrecisionMetric:
    """Evaluates whether the most relevant retrieved documents are ranked higher in the context list."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONTEXTUAL_PRECISION_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        context: list[str],
        query: str,
        expected: str,
    ) -> CheckResult:
        formatted_context = "\n\n".join(
            f"[Document {i}]:\n{doc}" for i, doc in enumerate(context, 1)
        )

        prompt = (
            f"User Query:\n{query}\n\n"
            f"Expected Answer:\n{expected}\n\n"
            f"Retrieved Context (ranked by retrieval order):\n{formatted_context}\n\n"
            f"LLM Output:\n{output}\n\n"
            "Assess the ranking quality of the retrieved context. "
            "Are the most relevant documents ranked higher? Score the contextual precision."
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
            metric_name="contextual_precision",
        )
