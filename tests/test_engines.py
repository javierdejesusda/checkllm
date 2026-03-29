"""Tests for checkllm.engines -- parallel evaluation engine system."""

from __future__ import annotations

import asyncio
import time

import pytest

from checkllm.engines import (
    AsyncEngine,
    BaseEngine,
    EngineStats,
    EngineType,
    HybridEngine,
    ProcessPoolEngine,
    ThreadPoolEngine,
    create_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _delay(seconds: float, value: int = 0) -> int:
    """Trivial async task that sleeps then returns *value*."""
    await asyncio.sleep(seconds)
    return value


def _cpu_square(n: int) -> int:
    """Pure function suitable for process-pool tests (must be picklable)."""
    return n * n


def _cpu_raise(x: int) -> int:
    """Top-level picklable function that always raises."""
    raise ValueError("process-boom")


# ---------------------------------------------------------------------------
# EngineStats
# ---------------------------------------------------------------------------

class TestEngineStats:
    def test_defaults(self):
        s = EngineStats()
        assert s.tasks_submitted == 0
        assert s.tasks_completed == 0
        assert s.total_execution_time == 0.0
        assert s.current_queue_depth == 0

    def test_average_execution_time_zero_completed(self):
        s = EngineStats()
        assert s.average_execution_time == 0.0

    def test_average_execution_time(self):
        s = EngineStats(tasks_completed=4, total_execution_time=2.0)
        assert s.average_execution_time == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EngineType enum
# ---------------------------------------------------------------------------

class TestEngineType:
    def test_values(self):
        assert EngineType.ASYNC == "async"
        assert EngineType.THREAD == "thread"
        assert EngineType.PROCESS == "process"
        assert EngineType.HYBRID == "hybrid"
        assert EngineType.AUTO == "auto"

    def test_from_string(self):
        assert EngineType("async") is EngineType.ASYNC
        assert EngineType("thread") is EngineType.THREAD


# ---------------------------------------------------------------------------
# AsyncEngine
# ---------------------------------------------------------------------------

class TestAsyncEngine:
    @pytest.mark.asyncio
    async def test_submit_and_gather(self):
        async with AsyncEngine(max_concurrency=5) as engine:
            tasks = [await engine.submit(_delay(0.01, i)) for i in range(5)]
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        engine = AsyncEngine(max_concurrency=3)
        tasks = [await engine.submit(_delay(0.01, i)) for i in range(6)]
        await engine.gather(tasks)
        await engine.shutdown()

        assert engine.stats.tasks_submitted == 6
        assert engine.stats.tasks_completed == 6
        assert engine.stats.average_execution_time > 0.0
        assert engine.stats.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_concurrency_limited(self):
        """With concurrency=2, tasks run in pairs, not all at once."""
        max_concurrent = 0
        current = 0

        async def _track() -> None:
            nonlocal max_concurrent, current
            current += 1
            if current > max_concurrent:
                max_concurrent = current
            await asyncio.sleep(0.05)
            current -= 1

        async with AsyncEngine(max_concurrency=2, max_queue_size=20) as engine:
            tasks = [await engine.submit(_track()) for _ in range(6)]
            await engine.gather(tasks)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_submits(self):
        engine = AsyncEngine()
        await engine.shutdown()
        coro = _delay(0.0)
        try:
            with pytest.raises(RuntimeError, match="shut-down"):
                await engine.submit(coro)
        finally:
            coro.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncEngine() as engine:
            assert isinstance(engine, BaseEngine)
        # After exiting, shutdown has been called.
        coro = _delay(0.0)
        try:
            with pytest.raises(RuntimeError):
                await engine.submit(coro)
        finally:
            coro.close()

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        async def _fail() -> None:
            raise ValueError("boom")

        async with AsyncEngine() as engine:
            task = await engine.submit(_fail())
            with pytest.raises(ValueError, match="boom"):
                await engine.gather([task])

    @pytest.mark.asyncio
    async def test_backpressure(self):
        """When max_queue_size is tiny, submit blocks until slots free up."""
        engine = AsyncEngine(max_concurrency=1, max_queue_size=2)

        submitted_count = 0

        async def _slow() -> int:
            await asyncio.sleep(0.05)
            return 1

        async def _submit_many() -> list[asyncio.Task[int]]:
            nonlocal submitted_count
            tasks: list[asyncio.Task[int]] = []
            for _ in range(4):
                t = await engine.submit(_slow())
                submitted_count += 1
                tasks.append(t)
            return tasks

        tasks = await _submit_many()
        results = await engine.gather(tasks)
        await engine.shutdown()

        assert len(results) == 4
        assert submitted_count == 4
        assert engine.stats.tasks_submitted == 4
        assert engine.stats.tasks_completed == 4

    @pytest.mark.asyncio
    async def test_repr(self):
        engine = AsyncEngine()
        r = repr(engine)
        assert "AsyncEngine" in r
        assert "submitted=" in r

    @pytest.mark.asyncio
    async def test_large_batch(self):
        async with AsyncEngine(max_concurrency=20, max_queue_size=200) as engine:
            tasks = [await engine.submit(_delay(0.001, i)) for i in range(50)]
            results = await engine.gather(tasks)
        assert len(results) == 50
        assert sorted(results) == list(range(50))


# ---------------------------------------------------------------------------
# ThreadPoolEngine
# ---------------------------------------------------------------------------

class TestThreadPoolEngine:
    @pytest.mark.asyncio
    async def test_submit_and_gather(self):
        async with ThreadPoolEngine(max_workers=3) as engine:
            tasks = [await engine.submit(_delay(0.01, i)) for i in range(5)]
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        engine = ThreadPoolEngine(max_workers=2)
        tasks = [await engine.submit(_delay(0.01, i)) for i in range(4)]
        await engine.gather(tasks)
        await engine.shutdown()

        assert engine.stats.tasks_submitted == 4
        assert engine.stats.tasks_completed == 4
        assert engine.stats.average_execution_time > 0.0
        assert engine.stats.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_runs_in_separate_threads(self):
        """Verify that work actually runs in non-main threads."""
        import threading

        main_thread = threading.current_thread().ident

        async def _get_thread_id() -> int:
            return threading.current_thread().ident  # type: ignore[return-value]

        async with ThreadPoolEngine(max_workers=2) as engine:
            tasks = [await engine.submit(_get_thread_id()) for _ in range(3)]
            thread_ids = await engine.gather(tasks)

        # At least some work should have run on a different thread.
        assert any(tid != main_thread for tid in thread_ids)

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        async def _fail() -> None:
            raise ValueError("thread-boom")

        async with ThreadPoolEngine(max_workers=1) as engine:
            task = await engine.submit(_fail())
            with pytest.raises(ValueError, match="thread-boom"):
                await engine.gather([task])

    @pytest.mark.asyncio
    async def test_context_manager_shutdown(self):
        async with ThreadPoolEngine(max_workers=1) as engine:
            task = await engine.submit(_delay(0.01, 42))
            results = await engine.gather([task])
        assert results == [42]

    @pytest.mark.asyncio
    async def test_repr(self):
        engine = ThreadPoolEngine(max_workers=2)
        assert "ThreadPoolEngine" in repr(engine)
        await engine.shutdown()


# ---------------------------------------------------------------------------
# ProcessPoolEngine
# ---------------------------------------------------------------------------

class TestProcessPoolEngine:
    @pytest.mark.asyncio
    async def test_submit_coroutine(self):
        """The submit() interface runs the coroutine on the main loop."""
        async with ProcessPoolEngine(max_workers=2) as engine:
            tasks = [await engine.submit(_delay(0.01, i)) for i in range(3)]
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_submit_func_cpu_bound(self):
        """submit_func dispatches a picklable function to a child process."""
        async with ProcessPoolEngine(max_workers=2) as engine:
            tasks = [await engine.submit_func(_cpu_square, i) for i in range(5)]
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 4, 9, 16]

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        engine = ProcessPoolEngine(max_workers=2)
        tasks = [await engine.submit_func(_cpu_square, i) for i in range(4)]
        await engine.gather(tasks)
        await engine.shutdown()

        assert engine.stats.tasks_submitted == 4
        assert engine.stats.tasks_completed == 4
        assert engine.stats.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        async with ProcessPoolEngine(max_workers=1) as engine:
            task = await engine.submit_func(_cpu_raise, 1)
            with pytest.raises(ValueError, match="process-boom"):
                await engine.gather([task])

    @pytest.mark.asyncio
    async def test_repr(self):
        engine = ProcessPoolEngine(max_workers=1)
        assert "ProcessPoolEngine" in repr(engine)
        await engine.shutdown()


# ---------------------------------------------------------------------------
# HybridEngine
# ---------------------------------------------------------------------------

class TestHybridEngine:
    @pytest.mark.asyncio
    async def test_default_submit_uses_async_engine(self):
        async with HybridEngine(max_concurrency=5) as engine:
            tasks = [await engine.submit(_delay(0.01, i)) for i in range(3)]
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2]
        # The async engine should have processed the tasks.
        assert engine.async_engine.stats.tasks_submitted == 3

    @pytest.mark.asyncio
    async def test_submit_io_routes_to_async(self):
        async with HybridEngine(max_concurrency=5) as engine:
            task = await engine.submit_io(_delay(0.01, 99))
            results = await engine.gather([task])
        assert results == [99]
        assert engine.async_engine.stats.tasks_submitted == 1
        assert engine.thread_engine.stats.tasks_submitted == 0

    @pytest.mark.asyncio
    async def test_submit_cpu_routes_to_thread(self):
        async with HybridEngine(max_concurrency=5, max_workers=2) as engine:
            task = await engine.submit_cpu(_delay(0.01, 77))
            results = await engine.gather([task])
        assert results == [77]
        assert engine.thread_engine.stats.tasks_submitted == 1

    @pytest.mark.asyncio
    async def test_submit_batch_below_threshold_uses_async(self):
        async with HybridEngine(routing_threshold=10) as engine:
            coros = [_delay(0.01, i) for i in range(5)]
            tasks = await engine.submit_batch(coros, cpu_bound=True)
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2, 3, 4]
        # Below threshold, so async engine is used.
        assert engine.async_engine.stats.tasks_submitted == 5
        assert engine.thread_engine.stats.tasks_submitted == 0

    @pytest.mark.asyncio
    async def test_submit_batch_above_threshold_uses_threads(self):
        async with HybridEngine(routing_threshold=3, max_workers=4) as engine:
            coros = [_delay(0.01, i) for i in range(5)]
            tasks = await engine.submit_batch(coros, cpu_bound=True)
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2, 3, 4]
        # Above threshold with cpu_bound=True, so thread engine is used.
        assert engine.thread_engine.stats.tasks_submitted == 5
        assert engine.async_engine.stats.tasks_submitted == 0

    @pytest.mark.asyncio
    async def test_submit_batch_io_ignores_threshold(self):
        """When cpu_bound=False, always use async engine regardless of size."""
        async with HybridEngine(routing_threshold=2) as engine:
            coros = [_delay(0.01, i) for i in range(5)]
            tasks = await engine.submit_batch(coros, cpu_bound=False)
            results = await engine.gather(tasks)
        assert sorted(results) == [0, 1, 2, 3, 4]
        assert engine.async_engine.stats.tasks_submitted == 5
        assert engine.thread_engine.stats.tasks_submitted == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        async with HybridEngine(max_concurrency=5, max_workers=2) as engine:
            io_tasks = [await engine.submit_io(_delay(0.01, i)) for i in range(3)]
            cpu_tasks = [await engine.submit_cpu(_delay(0.01, i)) for i in range(2)]
            await engine.gather(io_tasks + cpu_tasks)

        assert engine.stats.tasks_submitted == 5
        assert engine.stats.tasks_completed == 5
        assert engine.stats.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_shutdown_shuts_both_engines(self):
        engine = HybridEngine()
        await engine.submit(_delay(0.01, 1))
        await engine.shutdown()
        # Async engine should be shut down.
        coro = _delay(0.0)
        try:
            with pytest.raises(RuntimeError):
                await engine.async_engine.submit(coro)
        finally:
            coro.close()

    @pytest.mark.asyncio
    async def test_repr(self):
        engine = HybridEngine()
        assert "HybridEngine" in repr(engine)
        await engine.shutdown()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_async_engine_waits_for_pending(self):
        completed: list[int] = []

        async def _work(i: int) -> int:
            await asyncio.sleep(0.05)
            completed.append(i)
            return i

        engine = AsyncEngine(max_concurrency=5, max_queue_size=20)
        tasks = [await engine.submit(_work(i)) for i in range(5)]
        # Shutdown should wait for all tasks to complete.
        await engine.shutdown()
        assert len(completed) == 5

    @pytest.mark.asyncio
    async def test_thread_engine_waits_for_pending(self):
        engine = ThreadPoolEngine(max_workers=2)
        tasks = [await engine.submit(_delay(0.05, i)) for i in range(4)]
        await engine.shutdown()
        assert engine.stats.tasks_completed == 4

    @pytest.mark.asyncio
    async def test_process_engine_waits_for_pending(self):
        engine = ProcessPoolEngine(max_workers=2)
        tasks = [await engine.submit_func(_cpu_square, i) for i in range(4)]
        await engine.shutdown()
        assert engine.stats.tasks_completed == 4


# ---------------------------------------------------------------------------
# create_engine factory
# ---------------------------------------------------------------------------

class TestCreateEngine:
    def test_create_async(self):
        engine = create_engine("async", max_concurrency=5)
        assert isinstance(engine, AsyncEngine)

    def test_create_thread(self):
        engine = create_engine("thread", max_workers=2)
        assert isinstance(engine, ThreadPoolEngine)

    def test_create_process(self):
        engine = create_engine("process", max_workers=2)
        assert isinstance(engine, ProcessPoolEngine)

    def test_create_hybrid(self):
        engine = create_engine("hybrid", max_concurrency=5, max_workers=2)
        assert isinstance(engine, HybridEngine)

    def test_create_auto(self):
        engine = create_engine("auto")
        assert isinstance(engine, (AsyncEngine, HybridEngine))

    def test_create_from_enum(self):
        engine = create_engine(EngineType.ASYNC, max_concurrency=3)
        assert isinstance(engine, AsyncEngine)

    def test_create_invalid_type(self):
        with pytest.raises(ValueError):
            create_engine("nonexistent")

    @pytest.mark.asyncio
    async def test_factory_engine_is_functional(self):
        engine = create_engine("async", max_concurrency=3)
        async with engine:
            task = await engine.submit(_delay(0.01, 42))
            results = await engine.gather([task])
        assert results == [42]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_gather_empty_list(self):
        async with AsyncEngine() as engine:
            results = await engine.gather([])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_task(self):
        async with AsyncEngine() as engine:
            task = await engine.submit(_delay(0.0, 1))
            results = await engine.gather([task])
        assert results == [1]

    @pytest.mark.asyncio
    async def test_stats_after_exception(self):
        """Even if a task raises, stats should still be updated."""
        async def _fail() -> None:
            raise RuntimeError("test error")

        engine = AsyncEngine(max_concurrency=5, max_queue_size=10)
        task = await engine.submit(_fail())
        try:
            await engine.gather([task])
        except RuntimeError:
            pass
        await engine.shutdown()

        assert engine.stats.tasks_submitted == 1
        assert engine.stats.tasks_completed == 1
        assert engine.stats.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """gather with return_exceptions-like behavior via individual awaits."""
        async def _maybe_fail(i: int) -> int:
            if i == 2:
                raise ValueError("bad")
            return i

        async with AsyncEngine(max_concurrency=5, max_queue_size=10) as engine:
            tasks = [await engine.submit(_maybe_fail(i)) for i in range(5)]
            results: list[int | Exception] = []
            for t in tasks:
                try:
                    results.append(await t)
                except ValueError as e:
                    results.append(e)

        assert len(results) == 5
        assert isinstance(results[2], ValueError)
        assert results[0] == 0
        assert results[4] == 4
