"""Tests for checkllm.tracing — OpenTelemetry integration and lightweight tracing."""

from __future__ import annotations

import time

import pytest

from checkllm.models import CheckResult
from checkllm.tracing import Span, Tracer, get_tracer, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(passed: bool = True, name: str = "test") -> CheckResult:
    return CheckResult(
        passed=passed,
        score=0.9 if passed else 0.2,
        reasoning="ok" if passed else "fail",
        cost=0.001,
        latency_ms=50,
        metric_name=name,
    )


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_creates_span(self):
        now = time.time_ns()
        span = Span(
            name="test-span",
            start_time_ns=now,
            end_time_ns=now + 5_000_000,  # 5 ms
            attributes={"key": "value"},
            status="ok",
        )
        assert span.name == "test-span"
        assert span.start_time_ns == now
        assert span.end_time_ns == now + 5_000_000
        assert span.attributes == {"key": "value"}
        assert span.status == "ok"
        assert span.events == []
        assert span.children == []

    def test_duration_ms(self):
        span = Span(
            name="timed",
            start_time_ns=1_000_000_000,
            end_time_ns=1_010_000_000,  # 10 ms later
        )
        assert span.duration_ms == pytest.approx(10.0)

    def test_duration_ms_open_span(self):
        span = Span(name="open", start_time_ns=1_000_000_000, end_time_ns=0)
        assert span.duration_ms == 0.0

    def test_defaults(self):
        span = Span(name="minimal", start_time_ns=0)
        assert span.end_time_ns == 0
        assert span.attributes == {}
        assert span.events == []
        assert span.children == []
        assert span.status == "ok"


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class TestTracer:
    @pytest.fixture
    def tracer(self):
        return Tracer(service_name="test", enable_otel=False)

    def test_span_context_manager(self, tracer):
        with tracer.span("my-op", {"foo": "bar"}) as s:
            assert s.name == "my-op"
            assert s.attributes["foo"] == "bar"
            assert s.start_time_ns > 0

        # After exiting, end_time should be set
        assert s.end_time_ns > 0
        assert s.duration_ms >= 0  # may be 0.0 on fast machines / Windows
        assert s.status == "ok"

        # Span should appear in the export
        exported = tracer.export_json()
        assert len(exported) == 1
        assert exported[0]["name"] == "my-op"

    def test_span_records_error(self, tracer):
        with pytest.raises(ValueError, match="boom"):
            with tracer.span("failing-op") as s:
                raise ValueError("boom")

        assert s.status == "error"
        assert s.attributes["error.type"] == "ValueError"
        assert s.attributes["error.message"] == "boom"

    def test_nested_spans(self, tracer):
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass

        exported = tracer.export_json()
        assert len(exported) == 1  # Only one root span
        assert exported[0]["name"] == "parent"
        assert len(exported[0]["children"]) == 1
        assert exported[0]["children"][0]["name"] == "child"

    def test_trace_decorator_sync(self, tracer):
        @tracer.trace("sync-operation")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

        exported = tracer.export_json()
        assert len(exported) == 1
        assert exported[0]["name"] == "sync-operation"
        assert exported[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_trace_decorator_async(self, tracer):
        @tracer.trace("async-operation")
        async def my_async_function(x: int) -> int:
            return x + 1

        result = await my_async_function(10)
        assert result == 11

        exported = tracer.export_json()
        assert len(exported) == 1
        assert exported[0]["name"] == "async-operation"

    def test_trace_decorator_default_name(self, tracer):
        @tracer.trace()
        def plain_function():
            return 42

        plain_function()
        exported = tracer.export_json()
        assert len(exported) == 1
        # Default name is module.qualname
        assert "plain_function" in exported[0]["name"]

    def test_record_check_with_active_span(self, tracer):
        result = _make_result(passed=True, name="relevance")
        with tracer.span("eval") as s:
            tracer.record_check(result)

        # The check should be recorded as an event on the span
        assert len(s.events) == 1
        assert s.events[0]["name"] == "check.relevance"
        attrs = s.events[0]["attributes"]
        assert attrs["check.passed"] is True
        assert attrs["check.score"] == 0.9
        assert attrs["check.metric_name"] == "relevance"

    def test_record_check_without_active_span(self, tracer):
        result = _make_result(passed=False, name="hallucination")
        tracer.record_check(result)

        # Should create a standalone span
        exported = tracer.export_json()
        assert len(exported) == 1
        assert exported[0]["name"] == "check.hallucination"
        assert exported[0]["status"] == "error"  # because passed=False

    def test_export_json(self, tracer):
        with tracer.span("op1"):
            pass
        with tracer.span("op2"):
            pass

        exported = tracer.export_json()
        assert len(exported) == 2
        names = {e["name"] for e in exported}
        assert names == {"op1", "op2"}

        # Each entry should have required keys
        for entry in exported:
            assert "name" in entry
            assert "start_time_ns" in entry
            assert "end_time_ns" in entry
            assert "duration_ms" in entry
            assert "attributes" in entry
            assert "events" in entry
            assert "children" in entry
            assert "status" in entry

    def test_reset(self, tracer):
        with tracer.span("will-be-cleared"):
            pass

        assert len(tracer.export_json()) == 1
        tracer.reset()
        assert len(tracer.export_json()) == 0

    def test_otel_disabled(self):
        tracer = Tracer(enable_otel=False)
        assert tracer.has_otel is False


# ---------------------------------------------------------------------------
# Global tracer
# ---------------------------------------------------------------------------


class TestGlobalTracer:
    def test_get_tracer(self):
        t = get_tracer()
        assert isinstance(t, Tracer)
        # Should return the same instance on subsequent calls
        t2 = get_tracer()
        assert t is t2

    def test_trace_module_level_decorator(self):
        @trace("global-traced")
        def hello():
            return "world"

        result = hello()
        assert result == "world"
