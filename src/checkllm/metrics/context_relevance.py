from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are an expert context relevance evaluator for Retrieval-Augmented Generation (RAG) systems. Your job is to assess whether the retrieved context is relevant to the user's query — that is, whether the context contains information that would be useful for answering the query.

Score from 0.0 to 1.0:
- 1.0 = The context is highly relevant and contains all the information needed to fully answer the query.
- 0.8 = The context is mostly relevant with minor tangential content. It covers the main aspects of the query.
- 0.5 = The context is partially relevant. It addresses some aspects of the query but misses key parts, or contains significant irrelevant material.
- 0.3 = The context has only marginal relevance. Most of the content is unrelated to the query, with only superficial connections.
- 0.0 = The context is completely irrelevant to the query. It does not address the topic at all.

Key evaluation criteria:
1. Does the context directly address the topic of the query?
2. Does the context contain enough information to answer the query?
3. How much of the context is irrelevant noise vs. useful signal?
4. Would a human consider this context helpful for answering the query?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ContextRelevanceMetric:
    """Evaluates whether retrieved context is relevant to the query in a RAG pipeline."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONTEXT_RELEVANCE_SYSTEM_PROMPT

    async def evaluate(self, context: str, query: str) -> CheckResult:
        prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Is the retrieved context relevant to the query? "
            "Does it contain useful information for answering the query? Score it."
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
            metric_name="context_relevance",
        )
