"""Recall@k for retrieval ranking."""

from __future__ import annotations

import time
from collections.abc import Iterable

from checkllm.models import CheckResult


class RecallAtKMetric:
    """Fraction of relevant ids that appear in the top-``k`` retrieved list.

    Given a ranked list of retrieved context ids and a set of gold
    relevant ids, returns ``hits / |relevant|`` where ``hits`` is the
    number of unique relevant ids present in the top-``k``.
    """

    def __init__(self, k: int = 5, threshold: float = 0.5) -> None:
        """Initialize the metric.

        Args:
            k: Rank cutoff (must be positive). Recall is computed over the
                first ``k`` retrieved ids.
            threshold: Minimum recall required for ``passed`` to be True.

        Raises:
            ValueError: If ``k`` is not a positive integer, or if
                ``threshold`` is outside ``[0, 1]``.
        """
        if k <= 0:
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
        """Compute recall over the top-``k`` retrieved ids.

        Args:
            retrieved_ids: Ranked retrieved context ids (position 0 first).
            relevant_ids: Gold-relevant ids (binary relevance).

        Returns:
            CheckResult with Recall@k in ``[0, 1]``.
        """
        start = time.perf_counter_ns()

        relevant_set = set(relevant_ids)
        retrieved_at_k = retrieved_ids[: self.k]

        if not relevant_set:
            score = 0.0
            hits = 0
        else:
            seen: set[object] = set()
            hits = 0
            for doc_id in retrieved_at_k:
                if doc_id in relevant_set and doc_id not in seen:
                    seen.add(doc_id)
                    hits += 1
            score = hits / len(relevant_set)

        score = max(0.0, min(1.0, score))
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        reasoning = (
            f"Recall@{self.k}: {score:.4f} "
            f"({hits}/{len(relevant_set)} relevant ids found in top-{self.k})"
        )

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="recall_at_k",
        )
