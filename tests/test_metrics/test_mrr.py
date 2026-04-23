"""Tests for the MRR retrieval metric."""

from __future__ import annotations

import pytest

from checkllm.metrics.mrr import MRRMetric


class TestMRRMetric:
    @pytest.mark.asyncio
    async def test_perfect_ranking_top_hit(self) -> None:
        metric = MRRMetric(k=5, threshold=0.5)
        result = await metric.evaluate(["a", "b", "c"], {"a"})
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.metric_name == "mrr"

    @pytest.mark.asyncio
    async def test_worst_ranking_no_hit(self) -> None:
        metric = MRRMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["x", "y", "z"], {"a", "b"})
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_value_third_rank(self) -> None:
        metric = MRRMetric(k=5)
        result = await metric.evaluate(["x", "y", "a", "b"], {"a"})
        assert result.score == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_first_relevant_rank_used(self) -> None:
        metric = MRRMetric()
        # Two relevant ids; MRR is 1/rank of the first, not second.
        result = await metric.evaluate(["x", "a", "b"], {"a", "b"})
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_k_cutoff_excludes_later_hit(self) -> None:
        metric = MRRMetric(k=2)
        result = await metric.evaluate(["x", "y", "a"], {"a"})
        assert result.score == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            MRRMetric(k=-1)
