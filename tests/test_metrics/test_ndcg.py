"""Tests for the NDCG@k retrieval metric."""

from __future__ import annotations

import math

import pytest

from checkllm.metrics.ndcg import NDCGMetric


class TestNDCGMetric:
    @pytest.mark.asyncio
    async def test_perfect_ranking_binary(self) -> None:
        metric = NDCGMetric(k=3, threshold=0.5)
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = await metric.evaluate(retrieved, relevant)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.metric_name == "ndcg"

    @pytest.mark.asyncio
    async def test_worst_ranking_zero(self) -> None:
        metric = NDCGMetric(k=3, threshold=0.5)
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        result = await metric.evaluate(retrieved, relevant)
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_binary_value(self) -> None:
        # Retrieve [relevant, irrelevant, relevant]; relevant set has 2 items.
        # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3)     = 1 + 1/log2(3)
        metric = NDCGMetric(k=3)
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        result = await metric.evaluate(retrieved, relevant)
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert result.score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_graded_relevance_mapping(self) -> None:
        metric = NDCGMetric(k=3)
        retrieved = ["a", "b", "c"]
        gains = {"a": 3.0, "b": 2.0, "c": 1.0}
        # This is the ideal order, so NDCG must be 1.0.
        result = await metric.evaluate(retrieved, gains)
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_no_relevant_is_zero(self) -> None:
        metric = NDCGMetric(k=3)
        result = await metric.evaluate(["a", "b"], [])
        assert result.score == 0.0
        assert result.passed is False

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            NDCGMetric(k=0)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            NDCGMetric(threshold=1.5)
