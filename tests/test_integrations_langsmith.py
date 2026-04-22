"""Tests for the LangSmith tracer integration.

The ``langsmith`` SDK is mocked so these tests never touch the network.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from checkllm.models import CheckResult


class _FakeClient:
    """Minimal stand-in for ``langsmith.Client``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.runs: list[dict] = []
        self.updates: list[dict] = []
        self.feedback: list[dict] = []

    def create_run(self, **kwargs):
        self.runs.append(kwargs)

    def update_run(self, **kwargs):
        self.updates.append(kwargs)

    def create_feedback(self, **kwargs):
        self.feedback.append(kwargs)


@pytest.fixture
def fake_langsmith(monkeypatch):
    """Install a fake ``langsmith`` module into ``sys.modules``."""
    module = types.ModuleType("langsmith")
    module.Client = _FakeClient
    monkeypatch.setitem(sys.modules, "langsmith", module)
    # Make sure our integration picks the fake module up on a fresh import.
    import importlib

    import checkllm.integrations.langsmith as ls_module

    importlib.reload(ls_module)
    return module


def _make_check(metric: str = "relevance", passed: bool = True) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=0.9 if passed else 0.1,
        reasoning="because",
        cost=0.001,
        latency_ms=42,
        metric_name=metric,
    )


def test_import_error_when_sdk_missing(monkeypatch):
    """Constructor raises a helpful ImportError without ``langsmith``."""
    monkeypatch.setitem(sys.modules, "langsmith", None)
    from checkllm.integrations.langsmith import LangSmithTracer

    with pytest.raises(ImportError, match="langsmith"):
        LangSmithTracer()


def test_span_creates_and_updates_run(fake_langsmith):
    from checkllm.integrations.langsmith import LangSmithTracer

    client = _FakeClient()
    tracer = LangSmithTracer(client=client, project_name="proj")

    with tracer.span("evaluate", {"model": "gpt-4o"}):
        pass

    assert len(client.runs) == 1
    run = client.runs[0]
    assert run["name"] == "evaluate"
    assert run["project_name"] == "proj"
    assert run["inputs"] == {"model": "gpt-4o"}
    assert len(client.updates) == 1
    assert client.updates[0]["error"] is None


def test_span_records_error(fake_langsmith):
    from checkllm.integrations.langsmith import LangSmithTracer

    client = _FakeClient()
    tracer = LangSmithTracer(client=client)

    with pytest.raises(RuntimeError):
        with tracer.span("evaluate"):
            raise RuntimeError("boom")

    assert client.updates[0]["error"] == "boom"


def test_nested_spans_chain_parent_ids(fake_langsmith):
    from checkllm.integrations.langsmith import LangSmithTracer

    client = _FakeClient()
    tracer = LangSmithTracer(client=client)

    with tracer.span("outer"):
        with tracer.span("inner"):
            pass

    outer, inner = client.runs
    assert outer["parent_run_id"] is None
    assert inner["parent_run_id"] == outer["id"]


def test_record_check_emits_feedback(fake_langsmith):
    from checkllm.integrations.langsmith import LangSmithTracer

    client = _FakeClient()
    tracer = LangSmithTracer(client=client)

    with tracer.span("evaluate"):
        tracer.record_check(_make_check(passed=False))

    assert len(client.feedback) == 1
    fb = client.feedback[0]
    assert fb["key"] == "relevance"
    assert fb["score"] == pytest.approx(0.1)
    assert fb["value"] is False


def test_record_check_without_active_span_is_safe(fake_langsmith):
    from checkllm.integrations.langsmith import LangSmithTracer

    client = _FakeClient()
    tracer = LangSmithTracer(client=client)
    tracer.record_check(_make_check())
    assert client.feedback == []


def test_factory_returns_langsmith_tracer(fake_langsmith):
    from checkllm.integrations import get_tracer

    client = _FakeClient()
    tracer = get_tracer("langsmith", client=client)
    from checkllm.integrations.langsmith import LangSmithTracer

    assert isinstance(tracer, LangSmithTracer)
