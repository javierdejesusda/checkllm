from __future__ import annotations

import asyncio

import pytest

from checkllm.observe import (
    Span,
    Trace,
    clear_trace,
    get_trace,
    observe,
    start_trace,
    end_trace,
)


@pytest.fixture(autouse=True)
def _clean_trace():
    """Reset trace context before and after each test."""
    clear_trace()
    yield
    clear_trace()


class TestSpan:
    def test_duration_ms(self):
        span = Span(name="test", start_ms=1000, end_ms=1500)
        assert span.duration_ms == 500

    def test_duration_ms_zero_when_no_end(self):
        span = Span(name="test", start_ms=1000, end_ms=1000)
        assert span.duration_ms == 0

    def test_default_status_is_ok(self):
        span = Span(name="test")
        assert span.status == "ok"


class TestTrace:
    def test_empty_trace(self):
        trace = Trace()
        assert trace.root_span is None
        assert trace.total_latency_ms == 0
        assert trace.total_cost == 0.0

    def test_root_span(self):
        span = Span(name="root", start_ms=100, end_ms=200)
        trace = Trace(spans=[span])
        assert trace.root_span is not None
        assert trace.root_span.name == "root"

    def test_total_latency(self):
        spans = [
            Span(name="a", start_ms=100, end_ms=300),
            Span(name="b", start_ms=200, end_ms=500),
        ]
        trace = Trace(spans=spans)
        assert trace.total_latency_ms == 400

    def test_total_cost_recursive(self):
        child = Span(name="child", cost=0.01)
        parent = Span(name="parent", cost=0.02, children=[child])
        trace = Trace(spans=[parent])
        assert abs(trace.total_cost - 0.03) < 1e-10

    def test_to_dict(self):
        trace = Trace(trace_id="test_123", spans=[Span(name="s")])
        d = trace.to_dict()
        assert d["trace_id"] == "test_123"
        assert len(d["spans"]) == 1

    def test_to_json(self):
        trace = Trace(trace_id="test_456")
        j = trace.to_json()
        assert "test_456" in j


class TestObserveSync:
    def test_sync_function_traced(self):
        @observe(name="add", type="tool")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(3, 4)
        assert result == 7

        trace = get_trace()
        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "add"
        assert trace.spans[0].type == "tool"
        assert trace.spans[0].status == "ok"
        assert trace.spans[0].output_data == {"result": 7}

    def test_sync_captures_input(self):
        @observe(name="greet", capture_input=True)
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        greet("Alice")
        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].input_data["name"] == "Alice"

    def test_sync_no_capture_input(self):
        @observe(name="secret", capture_input=False)
        def secret(password: str) -> str:
            return "ok"

        secret("hunter2")
        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].input_data == {}

    def test_sync_no_capture_output(self):
        @observe(name="quiet", capture_output=False)
        def quiet() -> str:
            return "shh"

        quiet()
        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].output_data == {}

    def test_sync_error_handling(self):
        @observe(name="fail")
        def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].status == "error"
        assert "ValueError" in trace.spans[0].error

    def test_sync_default_name(self):
        @observe()
        def my_function() -> int:
            return 42

        my_function()
        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].name == "my_function"


class TestObserveAsync:
    @pytest.mark.asyncio
    async def test_async_function_traced(self):
        @observe(name="fetch", type="retriever")
        async def fetch(query: str) -> list[str]:
            return ["doc1", "doc2"]

        result = await fetch("test")
        assert result == ["doc1", "doc2"]

        trace = get_trace()
        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "fetch"
        assert trace.spans[0].type == "retriever"
        assert trace.spans[0].status == "ok"

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        @observe(name="async_fail")
        async def async_fail() -> None:
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            await async_fail()

        trace = get_trace()
        assert trace is not None
        assert trace.spans[0].status == "error"
        assert "RuntimeError" in trace.spans[0].error


class TestNestedSpans:
    @pytest.mark.asyncio
    async def test_nested_creates_parent_child(self):
        @observe(name="retrieve", type="retriever")
        async def retrieve(query: str) -> list[str]:
            return ["doc1"]

        @observe(name="generate", type="llm")
        async def generate(docs: list[str], query: str) -> str:
            return "answer"

        @observe(name="agent", type="agent")
        async def agent(query: str) -> str:
            docs = await retrieve(query)
            return await generate(docs, query)

        result = await agent("question")
        assert result == "answer"

        trace = get_trace()
        assert trace is not None
        assert len(trace.spans) == 1

        root = trace.spans[0]
        assert root.name == "agent"
        assert root.type == "agent"
        assert len(root.children) == 2
        assert root.children[0].name == "retrieve"
        assert root.children[0].type == "retriever"
        assert root.children[1].name == "generate"
        assert root.children[1].type == "llm"

    def test_nested_sync(self):
        @observe(name="inner", type="tool")
        def inner(x: int) -> int:
            return x * 2

        @observe(name="outer", type="agent")
        def outer(x: int) -> int:
            return inner(x) + 1

        result = outer(5)
        assert result == 11

        trace = get_trace()
        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "outer"
        assert len(trace.spans[0].children) == 1
        assert trace.spans[0].children[0].name == "inner"


class TestTraceContext:
    def test_get_trace_returns_none_when_empty(self):
        assert get_trace() is None

    def test_clear_trace_resets(self):
        @observe(name="op")
        def op() -> int:
            return 1

        op()
        assert get_trace() is not None

        clear_trace()
        assert get_trace() is None

    def test_manual_start_end_trace(self):
        trace = start_trace("my_trace")
        assert trace.trace_id.startswith("my_trace_")

        ended = end_trace()
        assert ended is trace
        assert get_trace() is trace

    @pytest.mark.asyncio
    async def test_concurrent_async_traces_isolated(self):
        """Verify that concurrent async tasks get independent traces."""
        results: dict[str, Trace | None] = {}

        @observe(name="task_a", type="tool")
        async def task_a() -> str:
            await asyncio.sleep(0.01)
            return "a"

        @observe(name="task_b", type="tool")
        async def task_b() -> str:
            await asyncio.sleep(0.01)
            return "b"

        async def run_a():
            await task_a()
            results["a"] = get_trace()

        async def run_b():
            await task_b()
            results["b"] = get_trace()

        await asyncio.gather(run_a(), run_b())

        trace_a = results.get("a")
        trace_b = results.get("b")
        assert trace_a is not None
        assert trace_b is not None


class TestTimingAndCost:
    @pytest.mark.asyncio
    async def test_span_has_timing(self):
        @observe(name="timed", type="llm")
        async def timed() -> str:
            await asyncio.sleep(0.01)
            return "done"

        await timed()
        trace = get_trace()
        assert trace is not None
        span = trace.spans[0]
        assert span.start_ms > 0
        assert span.end_ms >= span.start_ms
        assert span.duration_ms >= 0
