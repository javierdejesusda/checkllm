from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are an expert context relevance evaluator for Retrieval-Augmented Generation (RAG) systems. Your job is to assess precision: how much of the retrieved context is actually useful for answering the user's query, versus unrelated noise.

The score must reflect the ratio of query-relevant content to total context length. A verbose passage where only a few sentences matter is NOT a 1.0 — each irrelevant sentence is noise that dilutes the signal.

Score from 0.0 to 1.0 using this rubric:
- 1.0 = Every sentence in the context is directly on-topic for the query. There is no wasted content.
- 0.8 = The context is mostly on-topic; a small minority of sentences are tangential but not misleading.
- 0.6 = The context contains the answer, but roughly half the material is unrelated background or other topics.
- 0.4 = Only a small fraction of the context is relevant. Most sentences cover unrelated facts or digressions.
- 0.2 = The context barely touches the query topic. One or two sentences hint at it; the rest is off-topic.
- 0.0 = The context does not address the query at all.

Key evaluation steps:
1. Mentally segment the context into sentences or paragraphs.
2. Count how many of those units would directly help answer the query.
3. Compute the ratio (relevant units / total units) and map it to the rubric above.
4. Be willing to use intermediate values (0.55, 0.73, etc.) — do not cluster every answer at 1.0.

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
