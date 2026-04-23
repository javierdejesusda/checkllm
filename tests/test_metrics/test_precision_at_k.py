"""Tests for Precision@k."""

from __future__ import annotations

import pytest

from checkllm.metrics.precision_at_k import PrecisionAtKMetric


class TestPrecisionAtK:
    @pytest.mark.asyncio
    async def test_perfect_precision(self) -> None:
        metric = PrecisionAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["a", "b", "c"], {"a", "b", "c"})
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.metric_name == "precision_at_k"

    @pytest.mark.asyncio
    async def test_zero_precision(self) -> None:
        metric = PrecisionAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["x", "y", "z"], {"a", "b"})
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_value(self) -> None:
        metric = PrecisionAtKMetric(k=4)
        result = await metric.evaluate(["a", "x", "b", "y"], {"a", "b"})
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_duplicate_ids_counted_once(self) -> None:
        metric = PrecisionAtKMetric(k=3)
        result = await metric.evaluate(["a", "a", "x"], {"a"})
        assert result.score == pytest.approx(1.0 / 3.0)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            PrecisionAtKMetric(k=0)
