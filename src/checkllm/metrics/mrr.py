"""Mean Reciprocal Rank (MRR) at k.

Classical IR metric that scores a ranked retrieval by ``1 / rank`` of the
first relevant item. Deterministic and dependency-free.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from checkllm.models import CheckResult


class MRRMetric:
    """Compute the Reciprocal Rank of the first relevant id.

    Given a ranked list of retrieved context ids and a set of gold
    relevant ids, returns ``1 / rank`` where ``rank`` is the 1-based
    position of the first retrieved id that is in the relevant set. If no
    retrieved id is relevant (within the optional ``k`` cutoff), the score
    is ``0``.

    Calling this metric on a single example yields a reciprocal rank. The
    mean across a dataset is therefore MRR in the classical sense.

    This is a zero API cost alternative to LLM-judged retrieval quality.
    """

    def __init__(self, k: int | None = None, threshold: float = 0.5) -> None:
        """Initialize the metric.

        Args:
            k: Optional rank cutoff. Only the first ``k`` retrieved ids are
                considered. ``None`` scans the full retrieved list.
            threshold: Minimum reciprocal rank for ``passed`` to be True.

        Raises:
            ValueError: If ``k`` is provided and not a positive integer,
                or if ``threshold`` is outside ``[0, 1]``.
        """
        if k is not None and k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.k = k
        self.threshold = threshold

    async def evaluate(
        self,
        retrieved_ids: list[object],
        relevant_ids: Iterable[object],
    ) -> CheckResult:
        """Compute the reciprocal rank of the first relevant retrieved id.

        Args:
            retrieved_ids: Ranked retrieved context ids (position 0 is the
                top-ranked item).
            relevant_ids: Gold-relevant ids (binary relevance).

        Returns:
            CheckResult with the reciprocal rank in ``[0, 1]``.
        """
        start = time.perf_counter_ns()

        relevant_set = set(relevant_ids)
        cutoff = self.k if self.k is not None else len(retrieved_ids)
        retrieved_at_k = retrieved_ids[:cutoff]

        score = 0.0
        hit_rank = 0
        for rank, doc_id in enumerate(retrieved_at_k, start=1):
            if doc_id in relevant_set:
                score = 1.0 / rank
                hit_rank = rank
                break

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        k_label = str(self.k) if self.k is not None else "all"
        if hit_rank:
            reasoning = f"MRR@{k_label}: {score:.4f} " f"(first relevant id at rank {hit_rank})"
        else:
            reasoning = (
                f"MRR@{k_label}: 0.0000 " f"(no relevant id in first {len(retrieved_at_k)} results)"
            )

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="mrr",
        )
