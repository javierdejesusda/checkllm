"""Pre-built pytest fixtures for common checkllm patterns.

Import these in your conftest.py::

    from checkllm.pytest_helpers import shared_judge, budget_session
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from checkllm.check import CheckCollector
from checkllm.config import load_config

logger = logging.getLogger("checkllm.helpers")


@pytest.fixture(scope="session")
def shared_judge():
    """Session-scoped judge that reuses connections across all tests.

    Usage in conftest.py::

        from checkllm.pytest_helpers import shared_judge

        @pytest.fixture
        def check(shared_judge):
            config = load_config()
            return CheckCollector(config=config, judge=shared_judge)
    """
    config = load_config()
    collector = CheckCollector(config=config)
    judge = collector._get_judge()
    yield judge


@pytest.fixture(scope="session")
def budget_session():
    """Session-scoped budget tracker across all tests.

    Tracks total cost across the session and warns when approaching the limit.

    Usage in conftest.py::

        from checkllm.pytest_helpers import budget_session

    Then use it in tests::

        def test_my_llm(check, budget_session):
            budget_session.track(check)
            check.hallucination(output, context=ctx)
    """
    return _BudgetTracker()


class _BudgetTracker:
    """Tracks cumulative cost across a pytest session."""

    def __init__(self) -> None:
        config = load_config()
        self.budget = config.budget
        self.total_cost: float = 0.0
        self.check_count: int = 0

    def track(self, collector: CheckCollector) -> None:
        """Register a collector to track its costs after the test."""
        # Store reference — costs are tracked as checks run
        self._current_collector = collector

    def update(self) -> None:
        """Update totals from the current collector."""
        if hasattr(self, "_current_collector") and self._current_collector:
            for result in self._current_collector.results:
                self.total_cost += result.cost
                self.check_count += 1
            self._current_collector = None

    @property
    def remaining(self) -> float | None:
        """Remaining budget, or None if no budget set."""
        if self.budget is None:
            return None
        return max(0.0, self.budget - self.total_cost)

    @property
    def summary(self) -> str:
        """Human-readable cost summary."""
        if self.budget is not None:
            return (
                f"Session: {self.check_count} checks, "
                f"${self.total_cost:.4f} spent, "
                f"${self.remaining:.4f} remaining of ${self.budget:.2f}"
            )
        return f"Session: {self.check_count} checks, ${self.total_cost:.4f} spent"


@pytest.fixture(scope="session", autouse=False)
def auto_snapshot(request, tmp_path_factory):
    """Automatically save a snapshot after the pytest session.

    Usage in conftest.py::

        from checkllm.pytest_helpers import auto_snapshot
    """
    yield
    # After session: save snapshot if results exist
    try:
        from checkllm.pytest_plugin import get_session_results
        from checkllm.regression.snapshot import (
            MetricRecord,
            Snapshot,
            TestRunRecord,
            save_snapshot,
        )
        from datetime import datetime, timezone

        results = get_session_results()
        if not results:
            return

        tests = {}
        for node_id, checks in results.items():
            metrics = {}
            name_counts: dict[str, int] = {}
            for c in checks:
                count = name_counts.get(c.metric_name, 0)
                key = c.metric_name if count == 0 else f"{c.metric_name}_{count}"
                metrics[key] = MetricRecord(score=c.score, passed=c.passed)
                name_counts[c.metric_name] = count + 1
            tests[node_id] = [TestRunRecord(metrics=metrics)]

        snapshot_dir = Path(".checkllm/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = Snapshot(
            version=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tests=tests,
        )
        path = snapshot_dir / f"auto_{ts}.json"
        save_snapshot(snap, path)
        logger.info("Auto-saved snapshot to %s", path)
    except Exception as exc:
        logger.debug("Failed to auto-save snapshot: %s", exc)
