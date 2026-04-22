from __future__ import annotations

import functools
import inspect
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

from checkllm.check import CheckCollector
from checkllm.config import load_config
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
                    report.outcome = "failed"
                    report.longrepr = _format_check_failures(
                        item.nodeid,
                        failed,
                        collector.results,
                    )


def _format_check_failures(node_id, failed, all_results):
    """Build a rich, per-check failure report for the pytest terminal."""
    lines = [
        f"checkllm: {len(failed)}/{len(all_results)} check(s) failed",
        "",
    ]
    for r in failed:
        lines.append(r.format_failure())
        lines.append("")

    passed = [r for r in all_results if r.passed]
    if passed:
        names = ", ".join(r.metric_name for r in passed)
        lines.append(f"  Passed: {names}")

    return "\n".join(lines)


def pytest_configure(config: pytest.Config) -> None:
    """Reset session store and register checkllm markers.

    Args:
        config: The active pytest configuration.
    """
    global _store
    _store = _SessionStore()

    # Legacy short-name markers (kept for backward compatibility).
    config.addinivalue_line(
        "markers",
        "llm: mark test as requiring an LLM API key (deselect with '-m \"not llm\"')",
    )
    config.addinivalue_line(
        "markers",
        "deterministic: mark test as using only deterministic checks (no API calls)",
    )
    config.addinivalue_line(
        "markers",
        "expensive: mark test as high-cost (multiple LLM judge calls)",
    )
    config.addinivalue_line(
        "markers",
        "rag: mark test as evaluating RAG pipeline quality",
    )
    config.addinivalue_line(
        "markers",
        "safety: mark test as evaluating safety/toxicity/bias",
    )

    # Namespaced markers (preferred).
    for name, description in _CHECKLLM_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


# Markers registered by the plugin. Kept as a module constant so that
# ``pytest_configure`` and ``apply_checkllm_markers`` agree on names.
_CHECKLLM_MARKERS: dict[str, str] = {
    "checkllm_rag": "tests exercising RAG metrics (faithfulness, context relevance, etc.)",
    "checkllm_deterministic": "tests using only deterministic checks (no LLM calls)",
    "checkllm_llm": "tests using LLM judges (may cost money / hit network)",
    "checkllm_redteam": "security / red-team tests (prompt injection, jailbreak, PII)",
    "checkllm_multimodal": "vision / image / multimodal tests",
    "checkllm_slow": "tests taking >5s to execute",
    "checkllm_expensive": "tests costing >$0.10 estimated per run",
}


# Metric names that hit an LLM judge. Best-effort list — the auto
# detector is intentionally permissive.
_LLM_METRICS: frozenset[str] = frozenset(
    {
        "hallucination",
        "relevance",
        "toxicity",
        "rubric",
        "fluency",
        "coherence",
        "sentiment",
        "correctness",
        "faithfulness",
        "context_relevance",
        "answer_completeness",
        "instruction_following",
        "summarization",
        "bias",
        "consistency",
        "groundedness",
        "g_eval",
        "contextual_precision",
        "contextual_recall",
        "task_completion",
        "role_adherence",
        "tool_accuracy",
        "knowledge_retention",
        "conversation_completeness",
        "plan_quality",
        "goal_accuracy",
        "step_efficiency",
        "argument_correctness",
        "plan_adherence",
        "pii_detection",
        "misuse_detection",
        "role_violation",
        "non_advice",
        "image_coherence",
        "image_helpfulness",
        "image_relevance",
        "multimodal_faithfulness",
        "text_to_image",
        "mcp_task_completion",
        "mcp_use",
    }
)

_RAG_METRICS: frozenset[str] = frozenset(
    {
        "hallucination",
        "faithfulness",
        "context_relevance",
        "answer_completeness",
        "groundedness",
        "contextual_precision",
        "contextual_recall",
    }
)

_REDTEAM_METRICS: frozenset[str] = frozenset(
    {
        "toxicity",
        "bias",
        "pii_detection",
        "misuse_detection",
        "role_violation",
        "non_advice",
        "no_pii",
        "is_refusal",
    }
)

_MULTIMODAL_METRICS: frozenset[str] = frozenset(
    {
        "image_coherence",
        "image_helpfulness",
        "image_relevance",
        "multimodal_faithfulness",
        "text_to_image",
    }
)

_CHECK_CALL_RE = re.compile(r"\bcheck(?:\.expect|\.that\([^)]*\))?\s*\.\s*(a?\w+)\s*\(")


def _metric_names_in_source(source: str) -> set[str]:
    """Extract metric method names referenced via ``check.<name>(``.

    Async variants (``check.ahallucination``) and soft checks
    (``check.expect.rubric``) are normalised back to the base name.

    Args:
        source: Raw function source, possibly empty.

    Returns:
        Set of metric names discovered in ``source``.
    """
    if not source:
        return set()
    found: set[str] = set()
    for match in _CHECK_CALL_RE.finditer(source):
        name = match.group(1)
        if name.startswith("a") and name[1:] in _LLM_METRICS:
            name = name[1:]
        found.add(name)
    return found


def _existing_marker_names(item: pytest.Item) -> set[str]:
    """Return marker names already set on ``item``."""
    return {m.name for m in item.iter_markers()}


def apply_checkllm_markers(item: pytest.Item) -> Iterable[str]:
    """Auto-apply namespaced checkllm markers based on test source.

    The function reads the test's source (when available) and looks
    for ``check.<metric>()`` calls. Markers are added in-place via
    ``item.add_marker``; the function also returns the marker names
    that were added so callers can introspect or log.

    User-provided markers are never overwritten: if the test already
    carries a ``checkllm_*`` marker, no automatic marker is added.

    Args:
        item: The pytest item to annotate.

    Returns:
        Iterable of marker names that were added.
    """
    existing = _existing_marker_names(item)
    if any(name.startswith("checkllm_") for name in existing):
        return ()

    source = ""
    try:
        func = getattr(item, "function", None)
        if func is not None:
            source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""

    metrics = _metric_names_in_source(source)
    if not metrics:
        return ()

    added: list[str] = []
    hits_llm = bool(metrics & _LLM_METRICS)
    hits_rag = bool(metrics & _RAG_METRICS)
    hits_redteam = bool(metrics & _REDTEAM_METRICS)
    hits_multimodal = bool(metrics & _MULTIMODAL_METRICS)

    if hits_multimodal:
        item.add_marker("checkllm_multimodal")
        added.append("checkllm_multimodal")
    if hits_redteam:
        item.add_marker("checkllm_redteam")
        added.append("checkllm_redteam")
    if hits_rag:
        item.add_marker("checkllm_rag")
        added.append("checkllm_rag")
    if hits_llm:
        item.add_marker("checkllm_llm")
        added.append("checkllm_llm")
    elif metrics:
        # Only deterministic metrics detected.
        item.add_marker("checkllm_deterministic")
        added.append("checkllm_deterministic")
    return tuple(added)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply checkllm markers to every collected test item.

    Failures are swallowed so test collection can never be broken by
    the heuristic.
    """
    del config  # unused
    for item in items:
        try:
            apply_checkllm_markers(item)
        except Exception:
            # Best-effort, silent: never break collection.
            continue


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
        ids = [f"case-{i}-{c.query or c.input[:30]}" for i, c in enumerate(cases)]

        @pytest.mark.parametrize("case", cases, ids=ids)
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator
