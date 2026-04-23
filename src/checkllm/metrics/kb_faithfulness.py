"""Knowledge-base faithfulness metric.

Given an answer and a connected vector store, this metric pulls the top-k
contexts for a query, then delegates to
:class:`~checkllm.metrics.faithfulness.FaithfulnessMetric` to score whether
the answer stays grounded in the retrieved snippets. It is the "hallucination
against an external KB" check — identical in spirit to the standard
faithfulness metric but sources its context from a live knowledge store
rather than from a caller-supplied string.
"""

from __future__ import annotations

from typing import Any, Protocol

from checkllm.integrations.vectorstore_base import RetrievedContext
from checkllm.judge import JudgeBackend
from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.models import CheckResult


class _VectorStore(Protocol):
    """Minimal structural type for anything that can answer ``query``."""

    def query(
        self, vector_or_text: Any, top_k: int = 5, **kwargs: Any
    ) -> list[RetrievedContext]: ...


class KBFaithfulnessMetric:
    """Score an answer's faithfulness against an external knowledge base.

    The metric retrieves the top-k contexts for a query (using the connected
    store), joins them, and scores the answer against the joined context via
    the existing :class:`FaithfulnessMetric`.

    Args:
        judge: Judge backend used to score faithfulness.
        store: A connected vector-store connector (Pinecone, Weaviate,
            Milvus, Chroma, or anything satisfying the ``query`` protocol).
        top_k: Number of contexts to retrieve.
        threshold: Minimum faithfulness score for the check to pass.
        store_kwargs: Additional keyword arguments forwarded to ``store.query``.
    """

    metric_name: str = "kb_faithfulness"

    def __init__(
        self,
        judge: JudgeBackend,
        store: _VectorStore,
        top_k: int = 5,
        threshold: float = 0.8,
        store_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.judge = judge
        self.store = store
        self.top_k = top_k
        self.threshold = threshold
        self.store_kwargs = dict(store_kwargs or {})
        self._faithfulness = FaithfulnessMetric(judge=judge, threshold=threshold)

    def retrieve(self, query: str) -> list[RetrievedContext]:
        """Fetch the top-k contexts for ``query`` from the connected store."""
        hits = self.store.query(query, top_k=self.top_k, **self.store_kwargs)
        return list(hits)

    async def evaluate(
        self,
        output: str,
        query: str,
        *,
        contexts: list[RetrievedContext] | None = None,
    ) -> CheckResult:
        """Score ``output`` against contexts pulled from the KB for ``query``.

        Args:
            output: Model-generated answer.
            query: Original user query, used both to retrieve KB contexts and
                as additional grounding for the faithfulness judge.
            contexts: Optional pre-fetched contexts (bypasses retrieval).

        Returns:
            A :class:`CheckResult` named ``kb_faithfulness`` whose
            ``reasoning`` embeds the retrieved-context ids for audit.
        """
        hits = list(contexts) if contexts is not None else self.retrieve(query)
        joined = "\n\n".join(h.text for h in hits if h.text)

        if not joined.strip():
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=(
                    "No usable context retrieved from the knowledge base "
                    f"(hits={len(hits)}). Cannot assess faithfulness."
                ),
                cost=0.0,
                latency_ms=0,
                metric_name=self.metric_name,
                threshold=self.threshold,
                input_preview=output[:200],
            )

        inner = await self._faithfulness.evaluate(output=output, context=joined, query=query)
        ids = ", ".join(h.id for h in hits)
        reasoning = f"[kb_faithfulness] top_k={len(hits)} ids=[{ids}] | {inner.reasoning}"

        return CheckResult(
            passed=inner.passed,
            score=inner.score,
            reasoning=reasoning,
            cost=inner.cost,
            latency_ms=inner.latency_ms,
            metric_name=self.metric_name,
            threshold=self.threshold,
            input_preview=output[:200],
        )


__all__ = ["KBFaithfulnessMetric"]
