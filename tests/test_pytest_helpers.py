"""Tests for pytest helper fixtures."""

from checkllm.pytest_helpers import _BudgetTracker
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.testing import MockJudge


def test_budget_tracker_tracks_costs():
    tracker = _BudgetTracker()
    config = CheckllmConfig(budget=10.0)
    tracker.budget = config.budget
    tracker.total_cost = 0.5
    tracker.check_count = 3
    assert tracker.remaining == 9.5
    assert "3 checks" in tracker.summary
    assert "$0.5000" in tracker.summary


def test_budget_tracker_no_budget():
    tracker = _BudgetTracker()
    tracker.budget = None
    tracker.total_cost = 1.0
    tracker.check_count = 5
    assert tracker.remaining is None
    assert "$1.0000" in tracker.summary


def test_budget_tracker_track_collector():
    tracker = _BudgetTracker()
    config = CheckllmConfig()
    judge = MockJudge(default_score=0.9)
    collector = CheckCollector(config=config, judge=judge)
    collector.contains("hello world", "hello")
    tracker.track(collector)
    tracker.update()
    assert tracker.check_count == 1


def test_shared_judge_fixture_importable():
    """Verify the fixture function exists and is importable."""
    from checkllm.pytest_helpers import shared_judge

    assert callable(shared_judge)


def test_auto_snapshot_fixture_importable():
    from checkllm.pytest_helpers import auto_snapshot

    assert callable(auto_snapshot)
