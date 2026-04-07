"""Parallel evaluation engines for checkllm.

This module provides multiple execution strategies for running LLM evaluation
checks in parallel.  Each engine implements a common async interface and tracks
execution statistics (tasks submitted, completed, average execution time, queue
depth).

Engines
-------
- **AsyncEngine** -- asyncio semaphore-based concurrency with backpressure.
- **ThreadPoolEngine** -- wraps a ``ThreadPoolExecutor``; each worker thread
  runs its own event loop so async judge calls work transparently.
- **ProcessPoolEngine** -- wraps a ``ProcessPoolExecutor`` for CPU-bound
  deterministic checks at scale.
- **HybridEngine** -- routes I/O-bound judge calls to an ``AsyncEngine`` and
  CPU-bound deterministic checks to a ``ThreadPoolEngine``.

Use :func:`create_engine` to instantiate an engine by :class:`EngineType` name.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger("checkllm.engines")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class EngineStats:
    """Mutable counters shared by every engine instance."""

    tasks_submitted: int = 0
    tasks_completed: int = 0
    total_execution_time: float = 0.0
    current_queue_depth: int = 0

    @property
    def average_execution_time(self) -> float:
        """Mean wall-clock time per completed task (seconds)."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_execution_time / self.tasks_completed


# ---------------------------------------------------------------------------
# Engine type enum
# ---------------------------------------------------------------------------

class EngineType(str, enum.Enum):
    """Supported engine execution strategies."""

    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"
    HYBRID = "hybrid"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Abstract base engine
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    """Abstract base class for all evaluation engines."""

    def __init__(self) -> None:
        self._stats = EngineStats()

    # -- public stats access ------------------------------------------------

    @property
    def stats(self) -> EngineStats:
        return self._stats

    # -- abstract interface -------------------------------------------------

    @abstractmethod
    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Submit a coroutine for execution.

        Returns an :class:`asyncio.Task` (or compatible future) that can be
        passed to :meth:`gather`.
        """

    @abstractmethod
    async def gather(self, tasks: list[asyncio.Task[T]]) -> list[T]:
        """Wait for all *tasks* and return their results in order."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Perform a clean shutdown, waiting for any in-flight work."""

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> BaseEngine:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.shutdown()

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        s = self._stats
        return (
            f"{cls}(submitted={s.tasks_submitted}, "
            f"completed={s.tasks_completed}, "
            f"avg_time={s.average_execution_time:.4f}s, "
            f"queue={s.current_queue_depth})"
        )


# ---------------------------------------------------------------------------
# AsyncEngine
# ---------------------------------------------------------------------------

class AsyncEngine(BaseEngine):
    """Asyncio-based engine with semaphore concurrency and backpressure.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks executing simultaneously.
    max_queue_size:
        Maximum pending-task queue depth.  When the queue is full,
        :meth:`submit` will block (await) until a slot opens, providing
        natural backpressure to the caller.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        max_queue_size: int = 100,
    ) -> None:
        super().__init__()
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_queue_size = max_queue_size
        self._queue_semaphore = asyncio.Semaphore(max_queue_size)
        self._running_tasks: set[asyncio.Task[Any]] = set()
        self._shutdown_event = asyncio.Event()

    # -- internal wrapper ---------------------------------------------------

    async def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Wrap *coro* with semaphore gating, stats tracking, and cleanup."""
        start = time.monotonic()
        try:
            async with self._semaphore:
                return await coro
        finally:
            elapsed = time.monotonic() - start
            self._stats.tasks_completed += 1
            self._stats.total_execution_time += elapsed
            self._stats.current_queue_depth = max(
                0, self._stats.current_queue_depth - 1
            )
            # Release a backpressure slot so the next submit() can proceed.
            self._queue_semaphore.release()

    # -- public interface ---------------------------------------------------

    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot submit to a shut-down engine")

        # Backpressure: block until a queue slot opens.
        await self._queue_semaphore.acquire()

        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1

        task: asyncio.Task[T] = asyncio.create_task(self._run(coro))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)
        return task

    async def gather(self, tasks: list[asyncio.Task[T]]) -> list[T]:
        return list(await asyncio.gather(*tasks))

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        logger.debug("AsyncEngine shut down: %s", self._stats)


# ---------------------------------------------------------------------------
# ThreadPoolEngine
# ---------------------------------------------------------------------------

def _run_coro_in_new_loop(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async coroutine in a fresh event loop on the calling thread.

    We accept a *factory* (zero-arg callable returning a coroutine) rather
    than the coroutine itself because coroutines are not safe to transport
    across threads once created.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_factory())
    finally:
        loop.close()


class ThreadPoolEngine(BaseEngine):
    """Thread-based engine for running sync or async checks.

    Each worker thread creates its own event loop, making it safe to call
    async judge backends from synchronous code.

    Parameters
    ----------
    max_workers:
        Number of threads in the pool.  Defaults to ``min(32, cpu_count + 4)``
        following :class:`~concurrent.futures.ThreadPoolExecutor` conventions.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__()
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures: set[asyncio.Task[Any]] = set()

    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1

        loop = asyncio.get_running_loop()

        # Capture the already-created coroutine object.  Each coroutine is
        # used exactly once so this is safe even across threads.
        coro_ref = coro

        def _factory() -> Coroutine[Any, Any, T]:
            return coro_ref  # type: ignore[return-value]

        start = time.monotonic()

        future = loop.run_in_executor(self._pool, _run_coro_in_new_loop, _factory)

        async def _tracked() -> T:
            try:
                result = await future
                return result  # type: ignore[return-value]
            finally:
                elapsed = time.monotonic() - start
                self._stats.tasks_completed += 1
                self._stats.total_execution_time += elapsed
                self._stats.current_queue_depth = max(
                    0, self._stats.current_queue_depth - 1
                )

        task: asyncio.Task[T] = asyncio.create_task(_tracked())
        self._pending_futures.add(task)
        task.add_done_callback(self._pending_futures.discard)
        return task

    async def gather(self, tasks: list[asyncio.Task[T]]) -> list[T]:
        return list(await asyncio.gather(*tasks))

    async def shutdown(self) -> None:
        if self._pending_futures:
            await asyncio.gather(*self._pending_futures, return_exceptions=True)
        self._pool.shutdown(wait=True)
        logger.debug("ThreadPoolEngine shut down: %s", self._stats)


# ---------------------------------------------------------------------------
# ProcessPoolEngine
# ---------------------------------------------------------------------------

def _process_worker(func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
    """Top-level picklable function executed in a child process."""
    return func(*args, **kwargs)


class ProcessPoolEngine(BaseEngine):
    """Process-based engine for CPU-bound deterministic checks at scale.

    Serializes check *functions* and their arguments via the standard
    :mod:`concurrent.futures` machinery and dispatches them to worker
    processes.

    Parameters
    ----------
    max_workers:
        Number of worker processes.  Defaults to the number of CPUs.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__()
        self._max_workers = max_workers
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self._pending: set[asyncio.Task[Any]] = set()

    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Submit a coroutine for execution.

        .. note::

            For true cross-process execution of synchronous functions, prefer
            :meth:`submit_func`.  This method still runs the coroutine on the
            main event loop (since coroutines cannot be pickled), but it
            conforms to the :class:`BaseEngine` interface for interoperability.
        """
        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1
        start = time.monotonic()

        async def _wrapper() -> T:
            try:
                return await coro
            finally:
                elapsed = time.monotonic() - start
                self._stats.tasks_completed += 1
                self._stats.total_execution_time += elapsed
                self._stats.current_queue_depth = max(
                    0, self._stats.current_queue_depth - 1
                )

        task: asyncio.Task[T] = asyncio.create_task(_wrapper())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def submit_func(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Task[T]:
        """Submit a *synchronous* callable for execution in a child process.

        The function, positional args, and keyword args must all be
        picklable.  This is the preferred way to dispatch CPU-bound
        deterministic checks.
        """
        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1

        loop = asyncio.get_running_loop()
        start = time.monotonic()

        future = loop.run_in_executor(
            self._pool, _process_worker, func, args, kwargs
        )

        async def _tracked() -> T:
            try:
                result = await future
                return result  # type: ignore[return-value]
            finally:
                elapsed = time.monotonic() - start
                self._stats.tasks_completed += 1
                self._stats.total_execution_time += elapsed
                self._stats.current_queue_depth = max(
                    0, self._stats.current_queue_depth - 1
                )

        task: asyncio.Task[T] = asyncio.create_task(_tracked())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def gather(self, tasks: list[asyncio.Task[T]]) -> list[T]:
        return list(await asyncio.gather(*tasks))

    async def shutdown(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        self._pool.shutdown(wait=True)
        logger.debug("ProcessPoolEngine shut down: %s", self._stats)


# ---------------------------------------------------------------------------
# HybridEngine
# ---------------------------------------------------------------------------

class HybridEngine(BaseEngine):
    """Smart engine that routes work to the appropriate sub-engine.

    * I/O-bound coroutines (e.g. LLM judge calls) are routed to an
      :class:`AsyncEngine`.
    * When the number of CPU-bound tasks in a batch exceeds *routing_threshold*,
      they are routed to a :class:`ThreadPoolEngine` for true thread
      parallelism.

    Parameters
    ----------
    max_concurrency:
        Passed to the internal :class:`AsyncEngine`.
    max_workers:
        Passed to the internal :class:`ThreadPoolEngine`.
    routing_threshold:
        Minimum number of tasks before CPU-bound work is offloaded to the
        thread pool.  Below this threshold, everything runs on the
        :class:`AsyncEngine` to avoid thread-creation overhead.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        max_workers: int | None = None,
        routing_threshold: int = 10,
    ) -> None:
        super().__init__()
        self._async_engine = AsyncEngine(max_concurrency=max_concurrency)
        self._thread_engine = ThreadPoolEngine(max_workers=max_workers)
        self._routing_threshold = routing_threshold

    # -- routing helpers ----------------------------------------------------

    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Submit a coroutine to the async engine (default path)."""
        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1
        task = await self._async_engine.submit(coro)

        def _update_stats(t: asyncio.Task[Any]) -> None:
            self._stats.tasks_completed += 1
            self._stats.current_queue_depth = max(
                0, self._stats.current_queue_depth - 1
            )

        task.add_done_callback(_update_stats)
        return task

    async def submit_io(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Explicitly route an I/O-bound coroutine to the async engine."""
        return await self.submit(coro)

    async def submit_cpu(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Explicitly route a CPU-bound coroutine to the thread engine."""
        self._stats.tasks_submitted += 1
        self._stats.current_queue_depth += 1
        task = await self._thread_engine.submit(coro)

        def _update_stats(t: asyncio.Task[Any]) -> None:
            self._stats.tasks_completed += 1
            self._stats.current_queue_depth = max(
                0, self._stats.current_queue_depth - 1
            )

        task.add_done_callback(_update_stats)
        return task

    async def submit_batch(
        self,
        coroutines: list[Coroutine[Any, Any, T]],
        *,
        cpu_bound: bool = False,
    ) -> list[asyncio.Task[T]]:
        """Submit many coroutines at once with automatic routing.

        If *cpu_bound* is ``True`` **and** the batch size exceeds the routing
        threshold, work is sent to the thread pool.  Otherwise the async
        engine is used.
        """
        use_threads = cpu_bound and len(coroutines) >= self._routing_threshold
        engine = self._thread_engine if use_threads else self._async_engine

        tasks: list[asyncio.Task[T]] = []
        for coro in coroutines:
            self._stats.tasks_submitted += 1
            self._stats.current_queue_depth += 1
            task = await engine.submit(coro)

            def _update_stats(t: asyncio.Task[Any]) -> None:
                self._stats.tasks_completed += 1
                self._stats.current_queue_depth = max(
                    0, self._stats.current_queue_depth - 1
                )

            task.add_done_callback(_update_stats)
            tasks.append(task)
        return tasks

    async def gather(self, tasks: list[asyncio.Task[T]]) -> list[T]:
        return list(await asyncio.gather(*tasks))

    async def shutdown(self) -> None:
        await self._async_engine.shutdown()
        await self._thread_engine.shutdown()
        # Aggregate sub-engine timing into our own stats.
        self._stats.total_execution_time = (
            self._async_engine.stats.total_execution_time
            + self._thread_engine.stats.total_execution_time
        )
        logger.debug("HybridEngine shut down: %s", self._stats)

    @property
    def async_engine(self) -> AsyncEngine:
        """Access the internal async engine (for inspection / testing)."""
        return self._async_engine

    @property
    def thread_engine(self) -> ThreadPoolEngine:
        """Access the internal thread engine (for inspection / testing)."""
        return self._thread_engine


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _detect_engine_type() -> EngineType:
    """Heuristic auto-detection of the best engine type for the environment."""
    import os

    cpu_count = os.cpu_count() or 1
    if cpu_count >= 4:
        return EngineType.HYBRID
    return EngineType.ASYNC


def create_engine(engine_type: EngineType | str = EngineType.AUTO, **kwargs: Any) -> BaseEngine:
    """Factory function to create an engine by type.

    Parameters
    ----------
    engine_type:
        One of ``"async"``, ``"thread"``, ``"process"``, ``"hybrid"``, or
        ``"auto"``.  The string form and :class:`EngineType` enum are both
        accepted.
    **kwargs:
        Forwarded to the engine constructor (e.g. ``max_concurrency``,
        ``max_workers``, ``routing_threshold``).

    Returns
    -------
    BaseEngine
        An engine instance ready for use as an async context manager.
    """
    if isinstance(engine_type, str):
        engine_type = EngineType(engine_type)

    if engine_type is EngineType.AUTO:
        engine_type = _detect_engine_type()

    engines: dict[EngineType, type[BaseEngine]] = {
        EngineType.ASYNC: AsyncEngine,
        EngineType.THREAD: ThreadPoolEngine,
        EngineType.PROCESS: ProcessPoolEngine,
        EngineType.HYBRID: HybridEngine,
    }

    cls = engines[engine_type]
    return cls(**kwargs)
