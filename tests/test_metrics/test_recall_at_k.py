"""Tests for Recall@k."""

from __future__ import annotations

import pytest

from checkllm.metrics.recall_at_k import RecallAtKMetric


class TestRecallAtK:
    @pytest.mark.asyncio
    async def test_perfect_recall(self) -> None:
        metric = RecallAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["a", "b", "c"], {"a", "b"})
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.metric_name == "recall_at_k"

    @pytest.mark.asyncio
    async def test_zero_recall(self) -> None:
        metric = RecallAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["x", "y", "z"], {"a", "b"})
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_value_partial(self) -> None:
        metric = RecallAtKMetric(k=2)
        # Retrieve ["a","x"]; relevant {"a","b"}; recall = 1/2 = 0.5
        result = await metric.evaluate(["a", "x", "b"], {"a", "b"})
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_relevant_returns_zero(self) -> None:
        metric = RecallAtKMetric(k=3)
        result = await metric.evaluate(["a", "b"], [])
        assert result.score == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            RecallAtKMetric(k=0)
