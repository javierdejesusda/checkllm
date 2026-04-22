"""Tests for the LangFuse tracer integration.

The ``langfuse`` SDK is mocked; no network calls are made.
"""
from __future__ import annotations

import sys
import types

import pytest

from checkllm.models import CheckResult


class _FakeSpan:
    def __init__(self, name: str, metadata=None, parent=None):
        self.name = name
        self.metadata = metadata or {}
        self.parent = parent
        self.children: list[_FakeSpan] = []
        self.scores: list[dict] = []
        self.ended = False
        self.end_kwargs: dict | None = None

    def span(self, name, metadata=None):
        child = _FakeSpan(name, metadata, parent=self)
        self.children.append(child)
        return child

    def score(self, **kwargs):
        self.scores.append(kwargs)

    def end(self, **kwargs):
        self.ended = True
        self.end_kwargs = kwargs


class _FakeTrace(_FakeSpan):
    pass


class _FakeLangfuse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.traces: list[_FakeTrace] = []
        self.flushed = False

    def trace(self, id=None, name=None):
        t = _FakeTrace(name or "trace")
        self.traces.append(t)
        return t

    def flush(self):
        self.flushed = True


@pytest.fixture
def fake_langfuse(monkeypatch):
    module = types.ModuleType("langfuse")
    module.Langfuse = _FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    import importlib

    import checkllm.integrations.langfuse as lf_module

    importlib.reload(lf_module)
    return module


def _make_check() -> CheckResult:
    return CheckResult(
        passed=True,
        score=0.8,
        reasoning="ok",
        cost=0.002,
        latency_ms=50,
        metric_name="faithfulness",
    )


def test_import_error_when_sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)
    from checkllm.integrations.langfuse import LangFuseTracer

    with pytest.raises(ImportError, match="langfuse"):
        LangFuseTracer()


def test_span_creates_langfuse_span(fake_langfuse):
    from checkllm.integrations.langfuse import LangFuseTracer

    client = _FakeLangfuse()
    tracer = LangFuseTracer(client=client)

    with tracer.span("evaluate", {"k": "v"}):
        pass

    assert len(client.traces) == 1
    trace_obj = client.traces[0]
    assert len(trace_obj.children) == 1
    span = trace_obj.children[0]
    assert span.name == "evaluate"
    assert span.metadata == {"k": "v"}
    assert span.ended
    assert span.end_kwargs == {}


def test_span_records_error(fake_langfuse):
    from checkllm.integrations.langfuse import LangFuseTracer

    client = _FakeLangfuse()
    tracer = LangFuseTracer(client=client)

    with pytest.raises(ValueError):
        with tracer.span("evaluate"):
            raise ValueError("bad")

    span = client.traces[0].children[0]
    assert span.end_kwargs["level"] == "ERROR"
    assert span.end_kwargs["status_message"] == "bad"


def test_nested_spans_nest_in_langfuse(fake_langfuse):
    from checkllm.integrations.langfuse import LangFuseTracer

    client = _FakeLangfuse()
    tracer = LangFuseTracer(client=client)

    with tracer.span("outer"):
        with tracer.span("inner"):
            pass

    trace_obj = client.traces[0]
    assert len(trace_obj.children) == 1
    outer = trace_obj.children[0]
    assert outer.name == "outer"
    assert len(outer.children) == 1
    assert outer.children[0].name == "inner"


def test_record_check_scores_current_span(fake_langfuse):
    from checkllm.integrations.langfuse import LangFuseTracer

    client = _FakeLangfuse()
    tracer = LangFuseTracer(client=client)

    with tracer.span("evaluate"):
        tracer.record_check(_make_check())

    span = client.traces[0].children[0]
    assert span.scores == [
        {"name": "faithfulness", "value": pytest.approx(0.8), "comment": "ok"}
    ]


def test_flush_delegates_to_client(fake_langfuse):
    from checkllm.integrations.langfuse import LangFuseTracer

    client = _FakeLangfuse()
    tracer = LangFuseTracer(client=client)
    tracer.flush()
    assert client.flushed is True


def test_factory_returns_langfuse_tracer(fake_langfuse):
    from checkllm.integrations import get_tracer
    from checkllm.integrations.langfuse import LangFuseTracer

    tracer = get_tracer("langfuse", client=_FakeLangfuse())
    assert isinstance(tracer, LangFuseTracer)
