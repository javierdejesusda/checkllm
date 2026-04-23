"""Tests for the MAP@k retrieval metric."""

from __future__ import annotations

import pytest

from checkllm.metrics.map_at_k import MAPAtKMetric


class TestMAPAtKMetric:
    @pytest.mark.asyncio
    async def test_perfect_ranking(self) -> None:
        metric = MAPAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["a", "b", "c"], {"a", "b", "c"})
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.metric_name == "map_at_k"

    @pytest.mark.asyncio
    async def test_worst_ranking_zero(self) -> None:
        metric = MAPAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["x", "y", "z"], {"a", "b"})
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_value_alternating(self) -> None:
        # Retrieved ["a","x","b","y"]; relevant {"a","b"}; k=4.
        # Precisions at hits: rank1 = 1/1 = 1; rank3 = 2/3.
        # AP = (1 + 2/3) / min(4, 2) = (5/3) / 2 = 5/6.
        metric = MAPAtKMetric(k=4)
        result = await metric.evaluate(["a", "x", "b", "y"], {"a", "b"})
        assert result.score == pytest.approx(5.0 / 6.0)

    @pytest.mark.asyncio
    async def test_duplicate_relevant_id_counted_once(self) -> None:
        metric = MAPAtKMetric(k=3)
        result = await metric.evaluate(["a", "a", "b"], {"a", "b"})
        # Effective hits are unique: a@1, b@3 -> (1 + 2/3)/2 = 5/6
        assert result.score == pytest.approx(5.0 / 6.0)

    @pytest.mark.asyncio
    async def test_no_relevant_returns_zero(self) -> None:
        metric = MAPAtKMetric(k=3)
        result = await metric.evaluate(["a", "b"], [])
        assert result.score == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            MAPAtKMetric(k=0)
