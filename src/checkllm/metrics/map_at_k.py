"""Mean Average Precision (MAP) at k.

Computes Average Precision for a single ranked retrieval; the dataset
mean of these scores is MAP in the classical IR sense.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from checkllm.models import CheckResult


class MAPAtKMetric:
    """Compute Average Precision at rank ``k``.

    Average Precision (AP) is the mean of the precision values measured at
    each rank where a relevant id is retrieved, normalized by the total
    number of relevant ids (or by the number of relevant ids reachable
    inside the top-``k``, depending on the normalizer).

    This implementation follows the standard "AP@k" definition used by
    TREC and Ragas: the denominator is ``min(k, |relevant|)``, so a
    perfect retrieval inside the cutoff yields exactly ``1.0``.
    """

    def __init__(self, k: int | None = None, threshold: float = 0.5) -> None:
        """Initialize the metric.

        Args:
            k: Optional rank cutoff. ``None`` uses the full retrieved list.
            threshold: Minimum AP score required for ``passed`` to be True.

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
        """Compute Average Precision at ``k``.

        Args:
            retrieved_ids: Ranked retrieved context ids (position 0 first).
            relevant_ids: Gold-relevant ids (binary relevance).

        Returns:
            CheckResult with AP@k in ``[0, 1]``.
        """
        start = time.perf_counter_ns()

        relevant_set = set(relevant_ids)
        cutoff = self.k if self.k is not None else len(retrieved_ids)
        retrieved_at_k = retrieved_ids[:cutoff]

        if not relevant_set:
            score = 0.0
            hits = 0
        else:
            hits = 0
            precision_sum = 0.0
            seen: set[object] = set()
            for rank, doc_id in enumerate(retrieved_at_k, start=1):
                if doc_id in relevant_set and doc_id not in seen:
                    seen.add(doc_id)
                    hits += 1
                    precision_sum += hits / rank
            denominator = min(
                cutoff if self.k is not None else len(relevant_set),
                len(relevant_set),
            )
            score = precision_sum / denominator if denominator > 0 else 0.0

        score = max(0.0, min(1.0, score))
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        k_label = str(self.k) if self.k is not None else "all"
        reasoning = (
            f"MAP@{k_label}: {score:.4f} "
            f"({hits} relevant retrieved out of {len(relevant_set)} relevant, "
            f"{len(retrieved_at_k)} scanned)"
        )

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="map_at_k",
        )
