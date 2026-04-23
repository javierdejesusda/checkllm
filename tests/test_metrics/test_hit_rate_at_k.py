"""Tests for HitRate@k."""

from __future__ import annotations

import pytest

from checkllm.metrics.hit_rate_at_k import HitRateAtKMetric


class TestHitRateAtK:
    @pytest.mark.asyncio
    async def test_perfect_hit(self) -> None:
        metric = HitRateAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["a", "x", "y"], {"a"})
        assert result.score == 1.0
        assert result.passed is True
        assert result.metric_name == "hit_rate_at_k"

    @pytest.mark.asyncio
    async def test_miss_is_zero(self) -> None:
        metric = HitRateAtKMetric(k=3, threshold=0.5)
        result = await metric.evaluate(["x", "y", "z"], {"a", "b"})
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_known_value_cutoff_excludes_hit(self) -> None:
        metric = HitRateAtKMetric(k=2)
        # Relevant id is at rank 3, so a k=2 cutoff must miss.
        result = await metric.evaluate(["x", "y", "a"], {"a"})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_hit_within_cutoff(self) -> None:
        metric = HitRateAtKMetric(k=3)
        result = await metric.evaluate(["x", "y", "a"], {"a"})
        assert result.score == 1.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            HitRateAtKMetric(k=0)
