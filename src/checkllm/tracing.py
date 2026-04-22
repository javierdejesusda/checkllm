"""OpenTelemetry integration and lightweight tracing for checkllm.

Provides a ``Tracer`` that records evaluation spans and check results.
Integrates with OpenTelemetry if the ``opentelemetry`` package is installed,
otherwise stores spans locally for inspection and JSON export.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from pydantic import BaseModel, Field

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.tracing")

F = TypeVar("F", bound=Callable[..., Any])


class Span(BaseModel):
    """A single trace span representing a timed operation.

    Attributes
    ----------
    name:
        Human-readable name for this span (e.g. ``"evaluate"``, ``"judge.call"``).
    start_time_ns:
        Start time in nanoseconds since epoch.
    end_time_ns:
        End time in nanoseconds since epoch (0 while span is still open).
    attributes:
        Key-value metadata attached to the span.
    events:
        Timestamped events that occurred during the span.
    children:
        Nested child spans.
    status:
        ``"ok"`` or ``"error"``.
    """

    name: str
    start_time_ns: int
    end_time_ns: int = 0
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    children: list[Span] = Field(default_factory=list)
    status: str = "ok"  # "ok", "error"

    @property
    def duration_ms(self) -> float:
        """Duration of the span in milliseconds."""
        if self.end_time_ns == 0:
            return 0.0
        return (self.end_time_ns - self.start_time_ns) / 1_000_000


class Tracer:
    """Lightweight tracing for checkllm evaluation calls.

    Integrates with OpenTelemetry if available, otherwise stores spans
    locally for inspection and export.

    Parameters
    ----------
    service_name:
        The service name used for OpenTelemetry spans.
    enable_otel:
        If ``True``, attempt to import and use OpenTelemetry. Falls back
        to local-only tracing if the package is not installed.
    """

    def __init__(self, service_name: str = "checkllm", enable_otel: bool = True) -> None:
        self._service_name = service_name
        self._spans: list[Span] = []
        self._span_stack: list[Span] = []
        self._otel_tracer: Any = None

        if enable_otel:
            try:
                from opentelemetry import trace as otel_trace
                from opentelemetry.trace import StatusCode  # noqa: F401

                self._otel_tracer = otel_trace.get_tracer(service_name)
                self._otel_status_ok = StatusCode.OK
                self._otel_status_error = StatusCode.ERROR
                logger.debug("OpenTelemetry tracing enabled for '%s'", service_name)
            except ImportError:
                logger.debug("OpenTelemetry not installed; using local-only tracing")
                self._otel_tracer = None

    @property
    def has_otel(self) -> bool:
        """Whether OpenTelemetry is available and enabled."""
        return self._otel_tracer is not None

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[Span, None, None]:
        """Create a traced span as a context manager.

        Usage::

            with tracer.span("my_operation", {"key": "value"}) as s:
                # ... do work ...
                s.events.append({"name": "checkpoint", "timestamp_ns": time.time_ns()})

        Parameters
        ----------
        name:
            Human-readable span name.
        attributes:
            Optional key-value metadata to attach to the span.
        """
        local_span = Span(
            name=name,
            start_time_ns=time.time_ns(),
            attributes=dict(attributes) if attributes else {},
        )

        # Nest under parent span if one exists
        if self._span_stack:
            self._span_stack[-1].children.append(local_span)
        else:
            self._spans.append(local_span)

        self._span_stack.append(local_span)

        # OpenTelemetry span
        otel_span_ctx = None
        if self._otel_tracer is not None:
            otel_span_ctx = self._otel_tracer.start_as_current_span(
                name,
                attributes=self._sanitize_attributes(attributes) if attributes else None,
            )
            otel_span_ctx.__enter__()  # type: ignore[union-attr]

        try:
            yield local_span
            local_span.status = "ok"
            if otel_span_ctx is not None:
                from opentelemetry import trace as otel_trace

                current = otel_trace.get_current_span()
                current.set_status(self._otel_status_ok)
        except Exception as exc:
            local_span.status = "error"
            local_span.attributes["error.type"] = type(exc).__name__
            local_span.attributes["error.message"] = str(exc)
            if otel_span_ctx is not None:
                from opentelemetry import trace as otel_trace

                current = otel_trace.get_current_span()
                current.set_status(self._otel_status_error, str(exc))
                current.record_exception(exc)
            raise
        finally:
            local_span.end_time_ns = time.time_ns()
            self._span_stack.pop()
            if otel_span_ctx is not None:
                otel_span_ctx.__exit__(None, None, None)  # type: ignore[union-attr]

    def trace(self, name: str | None = None) -> Callable[[F], F]:
        """Decorator for tracing functions.

        Wraps the function in a span whose name defaults to the function's
        qualified name.

        Parameters
        ----------
        name:
            Optional span name. Defaults to ``module.function_name``.

        Usage::

            @tracer.trace()
            def my_function():
                ...

            @tracer.trace("custom-name")
            async def my_async_function():
                ...
        """

        def decorator(fn: F) -> F:
            span_name = name or f"{fn.__module__}.{fn.__qualname__}"

            if asyncio_iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(span_name):
                        return await fn(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]
            else:

                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(span_name):
                        return fn(*args, **kwargs)

                return sync_wrapper  # type: ignore[return-value]

        return decorator  # type: ignore[return-value]

    def record_check(self, result: CheckResult) -> None:
        """Record a check result as an event on the current span.

        If a span is currently active on the stack, the check result is
        added as an event. Otherwise, it is recorded as a top-level span.

        Parameters
        ----------
        result:
            The check result to record.
        """
        event = {
            "name": f"check.{result.metric_name}",
            "timestamp_ns": time.time_ns(),
            "attributes": {
                "check.metric_name": result.metric_name,
                "check.passed": result.passed,
                "check.score": result.score,
                "check.cost": result.cost,
                "check.latency_ms": result.latency_ms,
                "check.reasoning": result.reasoning,
            },
        }

        if self._span_stack:
            self._span_stack[-1].events.append(event)
        else:
            # No active span — create a standalone span for this check
            check_span = Span(
                name=f"check.{result.metric_name}",
                start_time_ns=time.time_ns(),
                end_time_ns=time.time_ns(),
                attributes={
                    "check.metric_name": result.metric_name,
                    "check.passed": result.passed,
                    "check.score": result.score,
                    "check.cost": result.cost,
                    "check.latency_ms": result.latency_ms,
                },
                events=[event],
                status="ok" if result.passed else "error",
            )
            self._spans.append(check_span)

        # Also record on OTel span if available
        if self._otel_tracer is not None:
            try:
                from opentelemetry import trace as otel_trace

                current = otel_trace.get_current_span()
                if current and current.is_recording():
                    current.add_event(
                        f"check.{result.metric_name}",
                        attributes=self._sanitize_attributes(event["attributes"]),
                    )
            except Exception:
                pass  # Don't let OTel errors break the evaluation

    def export_json(self) -> list[dict[str, Any]]:
        """Export all recorded spans as JSON-serializable dicts.

        Returns
        -------
        list[dict[str, Any]]
            List of span dictionaries, each containing nested children.
        """
        return [self._span_to_dict(s) for s in self._spans]

    def reset(self) -> None:
        """Clear all recorded spans and reset the tracer state."""
        self._spans.clear()
        self._span_stack.clear()

    @staticmethod
    def _span_to_dict(span: Span) -> dict[str, Any]:
        """Recursively convert a Span to a plain dict."""
        return {
            "name": span.name,
            "start_time_ns": span.start_time_ns,
            "end_time_ns": span.end_time_ns,
            "duration_ms": span.duration_ms,
            "attributes": span.attributes,
            "events": span.events,
            "children": [Tracer._span_to_dict(c) for c in span.children],
            "status": span.status,
        }

    @staticmethod
    def _sanitize_attributes(
        attrs: dict[str, Any] | None,
    ) -> dict[str, str | int | float | bool]:
        """Sanitize attributes for OpenTelemetry (only primitive types)."""
        if not attrs:
            return {}
        sanitized: dict[str, str | int | float | bool] = {}
        for k, v in attrs.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized


def asyncio_iscoroutinefunction(fn: Any) -> bool:
    """Check if a function is an async coroutine function.

    Handles both native async functions and functools.partial wrappers.
    """
    import asyncio
    import inspect

    if asyncio.iscoroutinefunction(fn):
        return True
    if inspect.iscoroutinefunction(fn):
        return True
    # Handle functools.partial
    if hasattr(fn, "func"):
        return asyncio.iscoroutinefunction(fn.func)
    return False


# ---------------------------------------------------------------------------
# Global tracer instance
# ---------------------------------------------------------------------------

_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance, creating one if necessary.

    Returns
    -------
    Tracer
        The global shared tracer.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def trace(name: str | None = None) -> Callable[[F], F]:
    """Module-level convenience decorator using the global tracer.

    Usage::

        @trace("my-operation")
        def my_function():
            ...

    Parameters
    ----------
    name:
        Optional span name. Defaults to the function's qualified name.
    """
    return get_tracer().trace(name)
