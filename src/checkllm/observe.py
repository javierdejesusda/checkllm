from __future__ import annotations

import asyncio
import contextvars
import functools
import time
import uuid
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field, computed_field

F = TypeVar("F", bound=Callable[..., Any])

SpanType = Literal["agent", "llm", "retriever", "tool", "custom"]
SpanStatus = Literal["ok", "error"]


class Span(BaseModel):
    """A single execution span within a trace.

    Args:
        name: Human-readable name for this span.
        type: Category of the span (agent, llm, retriever, tool, custom).
        start_ms: Start time in epoch milliseconds.
        end_ms: End time in epoch milliseconds.
        status: Whether the span completed successfully.
        input_data: Captured input arguments.
        output_data: Captured return value.
        children: Nested child spans.
        metadata: Arbitrary extra metadata.
        cost: Monetary cost of this span's execution.
        token_count: Number of tokens used, if applicable.
        error: Error message if status is "error".
    """

    name: str
    type: SpanType = "custom"
    start_ms: int = 0
    end_ms: int = 0
    status: SpanStatus = "ok"
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    children: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cost: float = 0.0
    token_count: int | None = None
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int:
        """Duration of the span in milliseconds."""
        return max(0, self.end_ms - self.start_ms)


class Trace(BaseModel):
    """A complete execution trace containing one or more spans.

    Args:
        trace_id: Unique identifier for this trace.
        spans: Top-level spans in the trace.
    """

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def root_span(self) -> Span | None:
        """The first top-level span, or None if the trace is empty."""
        return self.spans[0] if self.spans else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_latency_ms(self) -> int:
        """Total latency from the first span start to the last span end."""
        if not self.spans:
            return 0
        start = min(s.start_ms for s in self.spans)
        end = max(s.end_ms for s in self.spans)
        return max(0, end - start)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost(self) -> float:
        """Sum of costs across all spans (recursive)."""
        def _sum_cost(span: Span) -> float:
            return span.cost + sum(_sum_cost(c) for c in span.children)

        return sum(_sum_cost(s) for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Serialize the trace to a JSON string."""
        return self.model_dump_json(indent=2)


_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_current_trace", default=None
)
_span_stack: contextvars.ContextVar[list[Span]] = contextvars.ContextVar(
    "_span_stack", default=None  # type: ignore[arg-type]
)
_last_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_last_trace", default=None
)


def _get_span_stack() -> list[Span]:
    """Get or initialize the span stack for the current context."""
    stack = _span_stack.get(None)
    if stack is None:
        stack = []
        _span_stack.set(stack)
    return stack


def _epoch_ms() -> int:
    """Return the current time as epoch milliseconds."""
    return int(time.time() * 1000)


def start_trace(name: str | None = None) -> Trace:
    """Manually start a new trace context.

    Args:
        name: Optional name for the trace (used as trace_id prefix).

    Returns:
        The newly created Trace.
    """
    trace = Trace(trace_id=f"{name or 'trace'}_{uuid.uuid4().hex[:8]}")
    _current_trace.set(trace)
    _span_stack.set([])
    return trace


def end_trace() -> Trace | None:
    """End the current trace and return it.

    Returns:
        The completed Trace, or None if no trace was active.
    """
    trace = _current_trace.get(None)
    if trace is not None:
        _last_trace.set(trace)
        _current_trace.set(None)
        _span_stack.set([])
    return trace


def get_trace() -> Trace | None:
    """Get the current or most recently completed trace.

    Returns:
        The active Trace, or the last completed one, or None.
    """
    trace = _current_trace.get(None)
    if trace is not None:
        return trace
    return _last_trace.get(None)


def clear_trace() -> None:
    """Reset all trace context, clearing the current and last traces."""
    _current_trace.set(None)
    _last_trace.set(None)
    _span_stack.set([])


def _capture_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func: Callable[..., Any],
) -> dict[str, Any]:
    """Safely capture function arguments as a serializable dict.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.
        func: The function being called (for parameter name introspection).

    Returns:
        Dict of argument names to repr-safe values.
    """
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    captured: dict[str, Any] = {}
    for i, arg in enumerate(args):
        name = params[i] if i < len(params) else f"arg_{i}"
        captured[name] = _safe_repr(arg)
    for k, v in kwargs.items():
        captured[k] = _safe_repr(v)
    return captured


def _safe_repr(value: Any) -> Any:
    """Convert a value to a JSON-safe representation.

    Args:
        value: Any Python value.

    Returns:
        A string or primitive that is JSON-serializable.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            return f"[{type(value).__name__} of length {len(value)}]"
        return [_safe_repr(v) for v in value]
    if isinstance(value, dict):
        if len(value) > 10:
            return f"{{dict of length {len(value)}}}"
        return {str(k): _safe_repr(v) for k, v in value.items()}
    return repr(value)[:200]


def observe(
    name: str | None = None,
    type: SpanType = "custom",
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """Decorator factory that auto-captures execution traces from decorated functions.

    Creates a Span for each invocation, records timing and I/O, and nests
    child spans when decorated functions call other decorated functions.

    Works with both sync and async functions.

    Args:
        name: Span name. Defaults to the function name.
        type: Span type category.
        capture_input: Whether to capture function arguments.
        capture_output: Whether to capture the return value.

    Returns:
        A decorator that wraps the target function with tracing.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace.get(None)
            auto_started = False
            if trace is None:
                trace = start_trace(span_name)
                auto_started = True

            stack = _get_span_stack()

            span = Span(name=span_name, type=type, start_ms=_epoch_ms())
            if capture_input:
                span.input_data = _capture_args(args, kwargs, func)

            stack.append(span)

            try:
                result = await func(*args, **kwargs)
                if capture_output:
                    span.output_data = {"result": _safe_repr(result)}
                span.status = "ok"
                return result
            except Exception as exc:
                span.status = "error"
                span.error = f"{type.__class__.__name__ if not isinstance(type, str) else ''}{exc.__class__.__name__}: {exc}"
                span.error = f"{exc.__class__.__name__}: {exc}"
                raise
            finally:
                span.end_ms = _epoch_ms()
                stack.pop()

                if stack:
                    stack[-1].children.append(span)
                else:
                    trace.spans.append(span)

                if auto_started and not stack:
                    end_trace()

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace.get(None)
            auto_started = False
            if trace is None:
                trace = start_trace(span_name)
                auto_started = True

            stack = _get_span_stack()

            span = Span(name=span_name, type=type, start_ms=_epoch_ms())
            if capture_input:
                span.input_data = _capture_args(args, kwargs, func)

            stack.append(span)

            try:
                result = func(*args, **kwargs)
                if capture_output:
                    span.output_data = {"result": _safe_repr(result)}
                span.status = "ok"
                return result
            except Exception as exc:
                span.status = "error"
                span.error = f"{exc.__class__.__name__}: {exc}"
                raise
            finally:
                span.end_ms = _epoch_ms()
                stack.pop()

                if stack:
                    stack[-1].children.append(span)
                else:
                    trace.spans.append(span)

                if auto_started and not stack:
                    end_trace()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
