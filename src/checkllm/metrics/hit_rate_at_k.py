"""Hit Rate@k (a.k.a. Hit@k) for retrieval ranking."""

from __future__ import annotations

import time
from collections.abc import Iterable

from checkllm.models import CheckResult


class HitRateAtKMetric:
    """Binary indicator: did any relevant id appear in the top-``k``?

    Returns ``1.0`` if at least one of the top-``k`` retrieved ids is in
    the relevant set, otherwise ``0.0``. The dataset mean across examples
    is the classical Hit Rate @ k.
    """

    def __init__(self, k: int = 5, threshold: float = 0.5) -> None:
        """Initialize the metric.

        Args:
            k: Rank cutoff (must be positive).
            threshold: Minimum score for ``passed`` to be True. Since the
                per-example score is ``0`` or ``1``, any threshold in
                ``(0, 1]`` makes ``passed`` equivalent to "got a hit".

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
        """Return ``1.0`` if a relevant id is in the top-``k``, else ``0.0``.

        Args:
            retrieved_ids: Ranked retrieved context ids (position 0 first).
            relevant_ids: Gold-relevant ids (binary relevance).

        Returns:
            CheckResult with Hit Rate@k in ``{0.0, 1.0}``.
        """
        start = time.perf_counter_ns()

        relevant_set = set(relevant_ids)
        retrieved_at_k = retrieved_ids[: self.k]

        hit = any(doc_id in relevant_set for doc_id in retrieved_at_k)
        score = 1.0 if hit else 0.0

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        reasoning = (
            f"HitRate@{self.k}: {score:.1f} "
            f"({'hit' if hit else 'miss'}; {len(retrieved_at_k)} retrieved, "
            f"{len(relevant_set)} relevant)"
        )

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="hit_rate_at_k",
        )
