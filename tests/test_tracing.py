"""Tests for checkllm.tracing — OpenTelemetry integration and lightweight tracing."""

from __future__ import annotations

import re
import time

import pytest

from checkllm.models import CheckResult
from checkllm.tracing import Span, Tracer, get_tracer, propagate_trace_context, trace


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


# ---------------------------------------------------------------------------
# W3C trace context propagation
# ---------------------------------------------------------------------------


_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-(00|01)$")


class TestPropagateTraceContext:
    """Verify ``propagate_trace_context`` emits W3C-compliant headers."""

    def _install_tracer(self, monkeypatch: pytest.MonkeyPatch) -> Tracer:
        """Install a fresh local-only tracer as the global tracer."""
        import checkllm.tracing as tr_mod

        tracer = Tracer(enable_otel=False)
        monkeypatch.setattr(tr_mod, "_tracer", tracer)
        return tracer

    def test_returns_input_when_no_active_span(self, monkeypatch):
        self._install_tracer(monkeypatch)
        headers = propagate_trace_context({"Authorization": "Bearer x"})
        assert headers == {"Authorization": "Bearer x"}
        assert "traceparent" not in headers

    def test_none_input_returns_empty_when_no_span(self, monkeypatch):
        self._install_tracer(monkeypatch)
        headers = propagate_trace_context()
        assert headers == {}

    def test_emits_valid_traceparent_under_active_span(self, monkeypatch):
        tracer = self._install_tracer(monkeypatch)
        with tracer.span("eval"):
            headers = propagate_trace_context({"X-Auth": "abc"})
        # Original header preserved
        assert headers["X-Auth"] == "abc"
        # traceparent present and W3C-formatted
        traceparent = headers["traceparent"]
        assert _TRACEPARENT_RE.match(traceparent), f"bad format: {traceparent}"

    def test_traceparent_fields_match_active_span(self, monkeypatch):
        tracer = self._install_tracer(monkeypatch)
        with tracer.span("outer") as outer:
            with tracer.span("inner") as inner:
                headers = propagate_trace_context()

        parts = headers["traceparent"].split("-")
        assert len(parts) == 4
        version, trace_id, span_id, flags = parts
        assert version == "00"
        # Child trace_id inherits from parent so outer.trace_id == inner.trace_id
        assert trace_id == outer.trace_id == inner.trace_id
        # span id is the *active* span's id (the innermost one)
        assert span_id == inner.span_id
        assert flags == "01"  # sampled

    def test_child_spans_share_trace_id(self, monkeypatch):
        tracer = self._install_tracer(monkeypatch)
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass
        assert parent.trace_id == child.trace_id
        assert parent.span_id != child.span_id

    def test_does_not_overwrite_existing_traceparent(self, monkeypatch):
        tracer = self._install_tracer(monkeypatch)
        preset = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        with tracer.span("eval"):
            headers = propagate_trace_context({"traceparent": preset})
        assert headers["traceparent"] == preset

    def test_traceparent_regenerates_per_span_pair(self, monkeypatch):
        """Separate span stacks should produce different trace ids."""
        tracer = self._install_tracer(monkeypatch)
        with tracer.span("first"):
            first = propagate_trace_context()["traceparent"]
        with tracer.span("second"):
            second = propagate_trace_context()["traceparent"]
        assert first != second
