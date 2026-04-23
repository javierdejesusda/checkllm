"""In-process progress broker for streaming eval events.

This module decouples the check engine from any particular transport.
Anywhere in the engine run loop that wants to emit a lifecycle event calls
:func:`emit` on the global broker; subscribers (typically the WebSocket
dashboard) receive the event through an ``asyncio.Queue``.

Events are small JSON-serializable dicts with a mandatory ``type`` field.
Known event types:

* ``test_started`` -- fired when a test begins; payload ``{"test_id": str}``.
* ``check_completed`` -- fired after every individual check.
* ``test_completed`` -- fired when a test finishes.
* ``run_completed`` -- fired once per run when all tests are done.

The broker is intentionally lightweight (no external deps) so it can be
imported from code paths that may not have an event loop running.  Emission
is non-blocking: if no subscribers exist, the event is dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("checkllm.progress")


@dataclass
class ProgressEvent:
    """A single lifecycle event published by the broker."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            **self.payload,
        }


class ProgressBroker:
    """Publish/subscribe broker for progress events.

    Thread-safe.  Subscribers are bound to an ``asyncio`` loop; publishers
    may be synchronous or asynchronous and may run on other threads.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._subscribers: list[tuple[asyncio.Queue[ProgressEvent], asyncio.AbstractEventLoop]] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._history: list[ProgressEvent] = []

    # -- subscription -------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[ProgressEvent]:
        """Create a new subscriber queue bound to the running event loop.

        Returns:
            An ``asyncio.Queue`` that receives every subsequent event.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.append((queue, loop))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[ProgressEvent]) -> None:
        """Remove a previously-registered subscriber queue."""
        with self._lock:
            self._subscribers = [(q, loop) for (q, loop) in self._subscribers if q is not queue]

    # -- publication --------------------------------------------------------

    def emit(self, event_type: str, **payload: Any) -> ProgressEvent:
        """Publish a new event.

        Args:
            event_type: One of the documented event types.
            **payload: JSON-serializable keyword arguments merged into the
                event payload.

        Returns:
            The :class:`ProgressEvent` that was published.
        """
        event = ProgressEvent(type=event_type, payload=dict(payload))
        with self._lock:
            # Retain a short scrollback so late subscribers can bootstrap.
            self._history.append(event)
            if len(self._history) > self._maxsize:
                self._history = self._history[-self._maxsize :]
            subs = list(self._subscribers)

        for queue, loop in subs:
            try:
                # ``queue.put_nowait`` requires running on the queue's loop.
                # If we're already on that loop we can call it directly;
                # otherwise schedule thread-safely.
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    queue.put_nowait(event)
                else:
                    loop.call_soon_threadsafe(_safe_put_nowait, queue, event)
            except asyncio.QueueFull:
                logger.debug("progress broker dropped event (queue full): %s", event_type)
            except Exception:  # noqa: BLE001
                # A bad subscriber must never break emission.
                logger.debug("progress broker emit failed for %s", event_type, exc_info=True)
        return event

    # -- history ------------------------------------------------------------

    def history(self, last: int | None = None) -> list[ProgressEvent]:
        """Return recent events (chronological order)."""
        with self._lock:
            if last is None or last >= len(self._history):
                return list(self._history)
            return list(self._history[-last:])

    def clear_history(self) -> None:
        """Reset stored history (e.g. between tests)."""
        with self._lock:
            self._history.clear()

    # -- utilities ----------------------------------------------------------

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def _safe_put_nowait(queue: asyncio.Queue[ProgressEvent], event: ProgressEvent) -> None:
    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(event)


# ---------------------------------------------------------------------------
# Module-global broker (used by the dashboard + engine)
# ---------------------------------------------------------------------------

_broker: ProgressBroker = ProgressBroker()


def get_broker() -> ProgressBroker:
    """Return the process-wide :class:`ProgressBroker`."""
    return _broker


def emit(event_type: str, **payload: Any) -> ProgressEvent:
    """Shortcut for ``get_broker().emit(...)``."""
    return _broker.emit(event_type, **payload)


def reset_broker() -> None:
    """Replace the module-global broker with a fresh instance.

    Primarily useful in tests that want isolated history between cases.
    """
    global _broker
    _broker = ProgressBroker()


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def emit_test_started(test_id: str, **extra: Any) -> ProgressEvent:
    """Emit a ``test_started`` event."""
    return emit("test_started", test_id=test_id, **extra)


def emit_test_completed(
    test_id: str,
    *,
    passed: bool,
    duration_ms: float,
    checks: int = 0,
    cost: float = 0.0,
    **extra: Any,
) -> ProgressEvent:
    """Emit a ``test_completed`` event."""
    return emit(
        "test_completed",
        test_id=test_id,
        passed=bool(passed),
        duration_ms=float(duration_ms),
        checks=int(checks),
        cost=float(cost),
        **extra,
    )


def emit_check_completed(
    test_id: str,
    *,
    metric: str,
    passed: bool,
    score: float,
    cost: float = 0.0,
    duration_ms: float = 0.0,
    provider: str = "",
    model: str = "",
    **extra: Any,
) -> ProgressEvent:
    """Emit a ``check_completed`` event with summary metadata."""
    return emit(
        "check_completed",
        test_id=test_id,
        metric=metric,
        passed=bool(passed),
        score=float(score),
        cost=float(cost),
        duration_ms=float(duration_ms),
        provider=provider,
        model=model,
        **extra,
    )


def emit_run_completed(
    *,
    total_tests: int,
    total_checks: int,
    passed: int,
    failed: int,
    total_cost: float,
    duration_ms: float,
    **extra: Any,
) -> ProgressEvent:
    """Emit a ``run_completed`` event."""
    return emit(
        "run_completed",
        total_tests=int(total_tests),
        total_checks=int(total_checks),
        passed=int(passed),
        failed=int(failed),
        total_cost=float(total_cost),
        duration_ms=float(duration_ms),
        **extra,
    )


__all__ = [
    "ProgressBroker",
    "ProgressEvent",
    "emit",
    "emit_check_completed",
    "emit_run_completed",
    "emit_test_completed",
    "emit_test_started",
    "get_broker",
    "reset_broker",
]
