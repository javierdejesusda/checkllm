"""Comprehensive tests for checkllm.check — CheckCollector methods."""

from __future__ import annotations

import pytest

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.models import CheckFailedError, CheckResult
from checkllm.testing import MockJudge


def _c(budget: float | None = None, runs: int = 1) -> CheckCollector:
    config = CheckllmConfig(budget=budget, runs_per_test=runs)
    return CheckCollector(config=config)


def _c_with_judge(score: float = 0.9) -> CheckCollector:
    config = CheckllmConfig()
    judge = MockJudge(default_score=score)
    return CheckCollector(config=config, judge=judge)


class TestCheckCollectorProperties:
    def test_total_cost_starts_zero(self):
        c = _c()
        assert c.total_cost == 0.0

    def test_cache_stats_returns_dict(self):
        c = _c()
        stats = c.cache_stats
        assert isinstance(stats, dict)

    def test_repr_shows_passed_failed(self):
        c = _c()
        c.contains("Hello", "Hello")
        c.contains("Hello", "Goodbye")
        r = repr(c)
        assert "passed=1" in r
        assert "failed=1" in r
        assert "checks=2" in r

    def test_repr_shows_cost(self):
        c = _c()
        r = repr(c)
        assert "cost=" in r


class TestCheckCollectorExpect:
    def test_expect_returns_soft_proxy(self):
        from checkllm.expect import SoftCheckProxy

        c = _c()
        proxy = c.expect
        assert isinstance(proxy, SoftCheckProxy)

    def test_expect_same_instance_twice(self):
        c = _c()
        p1 = c.expect
        p2 = c.expect
        assert p1 is p2


class TestCheckCollectorTeardown:
    def test_teardown_passes_when_all_pass(self):
        c = _c()
        c.contains("Hello world", "Hello")
        c.contains("Hello world", "world")
        # Should not raise
        c.teardown()

    def test_teardown_raises_on_failure(self):
        c = _c()
        c.contains("Hello", "Goodbye")  # failing check
        with pytest.raises(CheckFailedError):
            c.teardown()

    def test_teardown_closes_cache(self):
        c = _c()
        c.contains("Hello", "Hello")
        c.teardown()
        # Should not raise

    def test_teardown_empty_results_passes(self):
        c = _c()
        c.teardown()  # no results, should pass


class TestCheckCollectorBudget:
    def test_check_budget_no_budget(self):
        c = _c(budget=None)
        assert c._check_budget() is True

    def test_check_budget_within_limit(self):
        c = _c(budget=5.0)
        c._accumulated_cost = 2.0
        assert c._check_budget() is True

    def test_check_budget_exceeded(self):
        c = _c(budget=5.0)
        c._accumulated_cost = 6.0
        assert c._check_budget() is False

    def test_budget_skip_creates_result(self):
        c = _c(budget=0.01)
        c._accumulated_cost = 100.0  # exceed budget
        result = c._make_budget_skip_result("hallucination")
        assert result.passed is True  # skipped = passes
        assert result.metric_name == "hallucination"
        assert "Skipped" in result.reasoning
        assert c._skipped_budget == 1


class TestCheckCollectorRunWithRepeats:
    def test_single_run_no_aggregation(self):
        c = _c_with_judge(0.9)

        # Use _run_with_repeats directly with a simple coroutine
        async def mock_coro():
            return CheckResult(
                passed=True,
                score=0.9,
                reasoning="ok",
                cost=0.001,
                latency_ms=10,
                metric_name="test",
            )

        result = c._run_with_repeats(mock_coro, runs=1)
        assert result.score == 0.9
        assert result.passed is True

    def test_multi_run_aggregation(self):
        c = _c_with_judge(0.9)
        call_count = 0

        async def mock_coro():
            nonlocal call_count
            call_count += 1
            score = 0.9 if call_count % 2 == 1 else 0.5
            return CheckResult(
                passed=score >= 0.7,
                score=score,
                reasoning="ok",
                cost=0.001,
                latency_ms=10,
                metric_name="test",
            )

        result = c._run_with_repeats(mock_coro, runs=4)
        assert result.metric_name == "test"
        assert 0.0 <= result.score <= 1.0
        # 4 runs called
        assert call_count == 4


class TestCheckCollectorTrackCost:
    def test_track_cost_zero_cost_no_change(self):
        c = _c()
        result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="ok",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
        )
        c._track_cost(result)
        assert c.total_cost == 0.0

    def test_track_cost_accumulates(self):
        c = _c()
        for _ in range(3):
            result = CheckResult(
                passed=True,
                score=0.9,
                reasoning="ok",
                cost=0.005,
                latency_ms=0,
                metric_name="test",
            )
            c._track_cost(result)
        assert abs(c.total_cost - 0.015) < 1e-10


class TestCheckCollectorThat:
    def test_that_returns_assertion_chain(self):
        from checkllm.chain import AssertionChain

        c = _c()
        chain = c.that("Hello world")
        assert isinstance(chain, AssertionChain)

    def test_that_integrates_with_check(self):
        c = _c()
        c.that("Python is great").contains("Python").not_contains("Java")
        assert len(c.results) == 2
        assert all(r.passed for r in c.results)
