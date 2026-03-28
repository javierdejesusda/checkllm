from __future__ import annotations

import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from checkllm.check import CheckCollector
from checkllm.config import load_config
from checkllm.datasets.case import Case
from checkllm.datasets.loader import load_dataset
from checkllm.models import CheckResult

_CHECKLLM_KEY = pytest.StashKey[CheckCollector]()


# ---------------------------------------------------------------------------
# Session-level result store
# ---------------------------------------------------------------------------

class _SessionStore:
    """Collects per-test CheckResults across the entire pytest session."""

    def __init__(self) -> None:
        self.results: dict[str, list[CheckResult]] = {}

    def record(self, node_id: str, results: list[CheckResult]) -> None:
        if results:
            self.results[node_id] = list(results)


_store = _SessionStore()


def get_session_results() -> dict[str, list[CheckResult]]:
    """Public accessor for session-wide results (used by CLI)."""
    return _store.results


# ---------------------------------------------------------------------------
# pytest command-line options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("checkllm", "checkllm LLM testing options")
    group.addoption(
        "--checkllm-snapshot",
        action="store",
        default=None,
        metavar="PATH",
        help="Save a checkllm snapshot to PATH after the session.",
    )
    group.addoption(
        "--checkllm-report",
        action="store",
        default=None,
        metavar="PATH",
        help="Generate a checkllm HTML report to PATH after the session.",
    )
    group.addoption(
        "--checkllm-junit",
        action="store",
        default=None,
        metavar="PATH",
        help="Generate a checkllm JUnit XML report to PATH after the session.",
    )
    group.addoption(
        "--checkllm-markdown",
        action="store",
        default=None,
        metavar="PATH",
        help="Generate a checkllm Markdown report to PATH after the session.",
    )
    group.addoption(
        "--checkllm-jsonl",
        action="store",
        default=None,
        metavar="PATH",
        help="Export checkllm results as JSONL to PATH after the session.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def check(request):
    """Provides a CheckCollector that collects results and raises on teardown."""
    config = load_config()
    collector = CheckCollector(config=config)
    request.node.stash[_CHECKLLM_KEY] = collector
    return collector


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail the test at call-report time if check collector has failures,
    and store results in the session store."""
    from checkllm.judge import JudgeConfigError

    outcome = yield
    report = outcome.get_result()

    # Convert JudgeConfigError (missing API key) to a skip
    if report.when == "call" and report.failed:
        if call.excinfo and call.excinfo.errisinstance(JudgeConfigError):
            report.outcome = "skipped"
            report.wasxfail = ""
            report.longrepr = (
                str(item.fspath),
                None,
                f"SKIPPED: {call.excinfo.value}",
            )
            return

    if report.when == "call":
        collector = item.stash.get(_CHECKLLM_KEY, None)
        if collector is not None and collector.results:
            # Record results in session store
            _store.record(item.nodeid, collector.results)
            # Fail the test if any checks failed
            if report.passed:
                failed = [r for r in collector.results if not r.passed]
                if failed:
                    count = len(failed)
                    names = ", ".join(r.metric_name for r in failed)
                    report.outcome = "failed"
                    report.longrepr = f"{count} check(s) failed: {names}"


def pytest_configure(config: pytest.Config) -> None:
    """Reset session store and register markers."""
    global _store
    _store = _SessionStore()

    config.addinivalue_line(
        "markers",
        "llm: mark test as requiring an LLM API key (deselect with '-m \"not llm\"')",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """After all tests, generate snapshots, reports, and record history."""
    config = session.config
    results = _store.results

    if not results:
        return

    snapshot_path = config.getoption("--checkllm-snapshot", default=None)
    if snapshot_path:
        try:
            _save_snapshot(results, Path(snapshot_path))
        except Exception as exc:
            import sys
            print(f"checkllm: failed to save snapshot: {exc}", file=sys.stderr)

    report_path = config.getoption("--checkllm-report", default=None)
    if report_path:
        try:
            _save_html_report(results, Path(report_path))
        except Exception as exc:
            import sys
            print(f"checkllm: failed to save HTML report: {exc}", file=sys.stderr)

    junit_path = config.getoption("--checkllm-junit", default=None)
    if junit_path:
        try:
            _save_junit_report(results, Path(junit_path))
        except Exception as exc:
            import sys
            print(f"checkllm: failed to save JUnit report: {exc}", file=sys.stderr)

    markdown_path = config.getoption("--checkllm-markdown", default=None)
    if markdown_path:
        try:
            from checkllm.reporting.markdown import generate_markdown_report
            generate_markdown_report(results, Path(markdown_path))
        except Exception as exc:
            import sys
            print(f"checkllm: failed to save Markdown report: {exc}", file=sys.stderr)

    jsonl_path = config.getoption("--checkllm-jsonl", default=None)
    if jsonl_path:
        try:
            from checkllm.reporting.jsonl import export_jsonl
            export_jsonl(results, Path(jsonl_path))
        except Exception as exc:
            import sys
            print(f"checkllm: failed to save JSONL export: {exc}", file=sys.stderr)

    # Record in history
    try:
        import os
        from checkllm.history import RunHistory
        label = os.environ.get("CHECKLLM_RUN_LABEL", "pytest")
        history = RunHistory()
        history.record_run(results, label=label)
        history.close()
    except Exception as exc:
        import sys
        print(f"checkllm: failed to record history: {exc}", file=sys.stderr)


def _save_snapshot(results: dict[str, list[CheckResult]], path: Path) -> None:
    from checkllm.regression.snapshot import (
        MetricRecord,
        Snapshot,
        TestRunRecord,
        save_snapshot,
    )

    tests: dict[str, list[TestRunRecord]] = {}
    for node_id, checks in results.items():
        # Use indexed keys to avoid collisions when the same metric runs twice
        metrics: dict[str, MetricRecord] = {}
        name_counts: dict[str, int] = {}
        for c in checks:
            count = name_counts.get(c.metric_name, 0)
            key = c.metric_name if count == 0 else f"{c.metric_name}_{count}"
            metrics[key] = MetricRecord(score=c.score, passed=c.passed)
            name_counts[c.metric_name] = count + 1
        runs = [TestRunRecord(metrics=metrics)]
        tests[node_id] = runs

    snap = Snapshot(
        version=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tests=tests,
    )
    save_snapshot(snap, path)


def _save_html_report(results: dict[str, list[CheckResult]], path: Path) -> None:
    from checkllm.reporting.html import generate_html_report

    generate_html_report(results, path)


def _save_junit_report(results: dict[str, list[CheckResult]], path: Path) -> None:
    from checkllm.reporting.junit import generate_junit_xml

    generate_junit_xml(results, path)


# ---------------------------------------------------------------------------
# Dataset decorator
# ---------------------------------------------------------------------------

def dataset(source: str | Path | Callable) -> Callable:
    """Decorator that parametrizes a test function over a dataset.

    Usage::

        @dataset("tests/datasets/cases.yaml")
        def test_my_agent(check, case):
            result = my_agent(case.input)
            check.hallucination(result, context=case.input)
    """

    def decorator(func: Callable) -> Callable:
        cases = load_dataset(source)
        ids = [
            f"case-{i}-{c.query or c.input[:30]}"
            for i, c in enumerate(cases)
        ]

        @pytest.mark.parametrize("case", cases, ids=ids)
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator
