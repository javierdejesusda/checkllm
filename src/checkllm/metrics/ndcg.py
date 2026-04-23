"""Normalized Discounted Cumulative Gain (NDCG) at k.

A classical information-retrieval ranking metric that rewards placing
relevant items near the top of a ranked list. Implemented without any
external dependencies (pure Python ``math.log2``) and without requiring
an LLM judge.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping

from checkllm.models import CheckResult


class NDCGMetric:
    """Compute Normalized Discounted Cumulative Gain at rank ``k``.

    The metric accepts a ranked list of retrieved context ids and either:

    * A collection of binary-relevant ids (treated as gain 1 each), or
    * A mapping from id to a non-negative gain (graded relevance).

    Gains are discounted by ``log2(rank + 1)`` and normalized against the
    ideal DCG. The resulting score lies in ``[0, 1]``.

    This is a zero API cost alternative to LLM-judged retrieval quality.
    """

    def __init__(self, k: int | None = None, threshold: float = 0.5) -> None:
        """Initialize the metric.

        Args:
            k: Rank cutoff. Only the first ``k`` retrieved ids are scored.
                ``None`` (default) scores the full retrieved list.
            threshold: Minimum NDCG required for ``passed`` to be True.

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

    @staticmethod
    def _gains_from(relevant: Iterable[object] | Mapping[object, float]) -> dict[object, float]:
        """Build a gain lookup from either a set of ids or a gain mapping."""
        if isinstance(relevant, Mapping):
            gains: dict[object, float] = {}
            for key, value in relevant.items():
                gain = float(value)
                if gain < 0.0:
                    raise ValueError(f"Relevance gain for id {key!r} must be >= 0, got {gain}")
                gains[key] = gain
            return gains
        return {item: 1.0 for item in relevant}

    @staticmethod
    def _dcg(gains: list[float]) -> float:
        """Compute Discounted Cumulative Gain for an ordered gain list."""
        return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))

    async def evaluate(
        self,
        retrieved_ids: list[object],
        relevant_ids: Iterable[object] | Mapping[object, float],
    ) -> CheckResult:
        """Compute NDCG@k for a ranked retrieval list.

        Args:
            retrieved_ids: Context ids in retrieval order (position 0 is the
                top-ranked item).
            relevant_ids: Either an iterable of gold-relevant ids (binary
                relevance) or a mapping from id to non-negative gain.

        Returns:
            CheckResult with the NDCG score in ``[0, 1]``.
        """
        start = time.perf_counter_ns()

        gain_lookup = self._gains_from(relevant_ids)
        cutoff = self.k if self.k is not None else len(retrieved_ids)
        retrieved_at_k = retrieved_ids[:cutoff]

        actual_gains = [gain_lookup.get(doc_id, 0.0) for doc_id in retrieved_at_k]
        ideal_gains_all = sorted(gain_lookup.values(), reverse=True)
        ideal_gains = ideal_gains_all[:cutoff]

        dcg = self._dcg(actual_gains)
        idcg = self._dcg(ideal_gains)

        if idcg <= 0.0:
            score = 0.0
        else:
            score = dcg / idcg
        score = max(0.0, min(1.0, score))

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        k_label = str(self.k) if self.k is not None else "all"
        reasoning = (
            f"NDCG@{k_label}: {score:.4f} "
            f"(DCG={dcg:.4f}, IDCG={idcg:.4f}, "
            f"retrieved={len(retrieved_ids)}, relevant={len(gain_lookup)})"
        )

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="ndcg",
        )
