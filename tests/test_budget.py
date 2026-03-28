"""Tests for cost budget enforcement."""
from unittest.mock import AsyncMock

import pytest

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.models import CheckResult, JudgeResponse


class TestCostBudget:
    def test_no_budget_allows_all(self):
        config = CheckllmConfig(budget=None, cache_enabled=False)
        collector = CheckCollector(config=config)
        assert collector._check_budget() is True

    def test_under_budget(self):
        config = CheckllmConfig(budget=10.0, cache_enabled=False)
        collector = CheckCollector(config=config)
        collector._accumulated_cost = 5.0
        assert collector._check_budget() is True

    def test_over_budget(self):
        config = CheckllmConfig(budget=1.0, cache_enabled=False)
        collector = CheckCollector(config=config)
        collector._accumulated_cost = 1.5
        assert collector._check_budget() is False

    def test_budget_skip_result(self):
        config = CheckllmConfig(budget=0.01, cache_enabled=False)
        collector = CheckCollector(config=config)
        collector._accumulated_cost = 0.02
        result = collector._make_budget_skip_result("hallucination")
        assert result.passed is True
        assert result.cost == 0.0
        assert "Skipped" in result.reasoning
        assert collector._skipped_budget == 1

    def test_cost_tracking(self):
        config = CheckllmConfig(cache_enabled=False)
        collector = CheckCollector(config=config)
        result = CheckResult(
            passed=True, score=0.9, reasoning="ok",
            cost=0.005, latency_ms=100, metric_name="test",
        )
        collector._track_cost(result)
        assert collector.total_cost == pytest.approx(0.005)
        collector._track_cost(result)
        assert collector.total_cost == pytest.approx(0.01)
