"""Additional tests for checkllm.tracing — covering uncovered branches."""

from __future__ import annotations

import time

import pytest

from checkllm.models import CheckResult
from checkllm.tracing import (
    Span,
    Tracer,
    asyncio_iscoroutinefunction,
    get_tracer,
)


def _make_result(passed: bool = True, name: str = "test") -> CheckResult:
    return CheckResult(
        passed=passed,
        score=0.9 if passed else 0.2,
        reasoning="ok" if passed else "fail",
        cost=0.001,
        latency_ms=50,
        metric_name=name,
    )


class TestAsyncioIscoroutinefunction:
    def test_sync_function_returns_false(self):
        def sync_fn():
            pass

        assert asyncio_iscoroutinefunction(sync_fn) is False

    def test_async_function_returns_true(self):
        async def async_fn():
            pass

        assert asyncio_iscoroutinefunction(async_fn) is True

    def test_partial_sync_returns_false(self):
        import functools

        def sync_fn(x):
            pass

        partial = functools.partial(sync_fn, 1)
        assert asyncio_iscoroutinefunction(partial) is False

    def test_partial_async_returns_true(self):
        import functools

        async def async_fn(x):
            pass

        partial = functools.partial(async_fn, 1)
        assert asyncio_iscoroutinefunction(partial) is True


class TestTracerSanitizeAttributes:
    def test_sanitize_empty(self):
        result = Tracer._sanitize_attributes(None)
        assert result == {}

    def test_sanitize_empty_dict(self):
        result = Tracer._sanitize_attributes({})
        assert result == {}

    def test_sanitize_primitives_unchanged(self):
        attrs = {"str": "hello", "int": 42, "float": 3.14, "bool": True}
        result = Tracer._sanitize_attributes(attrs)
        assert result["str"] == "hello"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True

    def test_sanitize_non_primitive_to_str(self):
        attrs = {"list": [1, 2, 3], "dict": {"nested": "value"}}
        result = Tracer._sanitize_attributes(attrs)
        assert isinstance(result["list"], str)
        assert isinstance(result["dict"], str)


class TestTracerSpanToDictRecursive:
    def test_span_to_dict_structure(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("parent") as parent:
            with tracer.span("child"):
                pass

        exported = tracer.export_json()
        assert len(exported) == 1
        parent_dict = exported[0]
        assert "children" in parent_dict
        assert len(parent_dict["children"]) == 1
        assert parent_dict["children"][0]["name"] == "child"

    def test_deeply_nested_spans(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("level1"):
            with tracer.span("level2"):
                with tracer.span("level3"):
                    pass

        exported = tracer.export_json()
        assert exported[0]["name"] == "level1"
        assert exported[0]["children"][0]["name"] == "level2"
        assert exported[0]["children"][0]["children"][0]["name"] == "level3"


class TestTracerMultipleTopLevelSpans:
    def test_multiple_root_spans(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("op1"):
            pass
        with tracer.span("op2"):
            pass
        with tracer.span("op3"):
            pass

        exported = tracer.export_json()
        assert len(exported) == 3
        names = [e["name"] for e in exported]
        assert "op1" in names
        assert "op2" in names
        assert "op3" in names


class TestTracerRecordCheckStandaloneSpan:
    def test_standalone_check_passed_status_ok(self):
        tracer = Tracer(enable_otel=False)
        result = _make_result(passed=True, name="safety")
        tracer.record_check(result)

        exported = tracer.export_json()
        assert len(exported) == 1
        assert exported[0]["status"] == "ok"
        assert exported[0]["name"] == "check.safety"

    def test_standalone_check_failed_status_error(self):
        tracer = Tracer(enable_otel=False)
        result = _make_result(passed=False, name="hallucination")
        tracer.record_check(result)

        exported = tracer.export_json()
        assert exported[0]["status"] == "error"

    def test_standalone_check_attributes(self):
        tracer = Tracer(enable_otel=False)
        result = _make_result(passed=True, name="relevance")
        tracer.record_check(result)

        exported = tracer.export_json()
        attrs = exported[0]["attributes"]
        assert attrs["check.metric_name"] == "relevance"
        assert attrs["check.passed"] is True
        assert "check.score" in attrs


class TestTracerSpanDurationMs:
    def test_duration_ms_nonzero_after_close(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("timed") as s:
            time.sleep(0.001)

        assert s.duration_ms > 0

    def test_duration_ms_zero_for_open_span(self):
        span = Span(name="open", start_time_ns=1_000_000_000, end_time_ns=0)
        assert span.duration_ms == 0.0


class TestTracerSpanEvents:
    def test_events_accumulated_in_span(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("op") as s:
            s.events.append({"name": "custom_event", "timestamp_ns": time.time_ns()})

        assert len(s.events) == 1
        assert s.events[0]["name"] == "custom_event"

    def test_check_event_added_to_active_span(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("evaluation") as s:
            tracer.record_check(_make_result(passed=True, name="safety"))
            tracer.record_check(_make_result(passed=False, name="pii"))

        assert len(s.events) == 2
        event_names = [e["name"] for e in s.events]
        assert "check.safety" in event_names
        assert "check.pii" in event_names


class TestTracerResetAndBetweenTests:
    def test_reset_clears_spans_and_stack(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("s1"):
            pass
        with tracer.span("s2"):
            pass

        assert len(tracer.export_json()) == 2
        tracer.reset()
        assert len(tracer.export_json()) == 0
        assert tracer._span_stack == []

    def test_spans_accumulate_across_calls(self):
        tracer = Tracer(enable_otel=False)
        for i in range(5):
            with tracer.span(f"span_{i}"):
                pass

        assert len(tracer.export_json()) == 5


class TestTracerSpanWithAttributes:
    def test_span_with_none_attributes(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("no-attrs", None) as s:
            pass
        assert s.attributes == {}

    def test_span_with_attributes_dict(self):
        tracer = Tracer(enable_otel=False)
        with tracer.span("with-attrs", {"model": "gpt-4o", "tokens": 100}) as s:
            pass
        assert s.attributes["model"] == "gpt-4o"
        assert s.attributes["tokens"] == 100


class TestGetTracer:
    def test_returns_same_instance(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_returns_tracer_instance(self):
        t = get_tracer()
        assert isinstance(t, Tracer)


class TestTraceDecorator:
    def test_trace_wraps_function(self):
        tracer = Tracer(enable_otel=False)

        @tracer.trace("my_op")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_trace_default_name(self):
        tracer = Tracer(enable_otel=False)

        @tracer.trace()
        def compute_sum(x):
            return x * 2

        compute_sum(5)
        exported = tracer.export_json()
        assert any("compute_sum" in e["name"] for e in exported)

    @pytest.mark.asyncio
    async def test_trace_async_function(self):
        tracer = Tracer(enable_otel=False)

        @tracer.trace("async_op")
        async def fetch(url: str) -> str:
            return f"response from {url}"

        result = await fetch("https://example.com")
        assert "response from" in result
        exported = tracer.export_json()
        assert any(e["name"] == "async_op" for e in exported)

    def test_trace_captures_error(self):
        tracer = Tracer(enable_otel=False)

        @tracer.trace("risky_op")
        def risky():
            raise RuntimeError("something broke")

        with pytest.raises(RuntimeError, match="something broke"):
            risky()

        exported = tracer.export_json()
        assert exported[0]["status"] == "error"
        assert exported[0]["attributes"].get("error.type") == "RuntimeError"
