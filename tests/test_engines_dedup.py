"""Integration tests for AsyncEngine with in-flight deduplication."""

from __future__ import annotations

import asyncio

import pytest

from checkllm.dedup import make_dedup_key
from checkllm.engines import AsyncEngine


class _Counter:
    def __init__(self, delay: float = 0.03) -> None:
        self.calls = 0
        self.delay = delay

    async def work(self, tag: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return f"result:{tag}"


class TestAsyncEngineDedup:
    @pytest.mark.asyncio
    async def test_dedup_on_by_default(self) -> None:
        engine = AsyncEngine(max_concurrency=8)
        try:
            assert engine.deduplicator is not None
        finally:
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_dedup_off_when_disabled(self) -> None:
        engine = AsyncEngine(max_concurrency=8, dedup=False)
        try:
            assert engine.deduplicator is None
        finally:
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_submit_dedup_coalesces_same_key(self) -> None:
        engine = AsyncEngine(max_concurrency=16)
        counter = _Counter()
        key = make_dedup_key("judge", "same-prompt", model="m")

        try:
            tasks = [await engine.submit_dedup(key, lambda: counter.work("p")) for _ in range(10)]
            results = await engine.gather(tasks)
        finally:
            await engine.shutdown()

        assert all(r == "result:p" for r in results)
        assert counter.calls == 1
        assert engine.deduplicator.hits == 9
        assert engine.deduplicator.misses == 1

    @pytest.mark.asyncio
    async def test_submit_dedup_independent_for_different_keys(self) -> None:
        engine = AsyncEngine(max_concurrency=16)
        counter = _Counter()

        try:
            tasks = []
            for tag in ("a", "b", "c"):
                key = make_dedup_key("judge", tag)
                tasks.append(await engine.submit_dedup(key, lambda t=tag: counter.work(t)))
            results = await engine.gather(tasks)
        finally:
            await engine.shutdown()

        assert set(results) == {"result:a", "result:b", "result:c"}
        assert counter.calls == 3
        assert engine.deduplicator.misses == 3

    @pytest.mark.asyncio
    async def test_submit_dedup_disabled_runs_every_call(self) -> None:
        engine = AsyncEngine(max_concurrency=16, dedup=False)
        counter = _Counter()
        key = make_dedup_key("judge", "x")

        try:
            tasks = [await engine.submit_dedup(key, lambda: counter.work("x")) for _ in range(5)]
            results = await engine.gather(tasks)
        finally:
            await engine.shutdown()

        assert len(results) == 5
        assert counter.calls == 5

    @pytest.mark.asyncio
    async def test_submit_dedup_respects_concurrency(self) -> None:
        # Different keys must still obey the semaphore.
        engine = AsyncEngine(max_concurrency=2)
        active = 0
        peak = 0

        async def job() -> int:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return 1

        try:
            tasks = []
            for i in range(8):
                key = make_dedup_key("j", f"p-{i}")
                tasks.append(await engine.submit_dedup(key, job))
            results = await engine.gather(tasks)
        finally:
            await engine.shutdown()

        assert sum(results) == 8
        assert peak <= 2

    @pytest.mark.asyncio
    async def test_submit_dedup_stats_tracked(self) -> None:
        engine = AsyncEngine(max_concurrency=8)
        counter = _Counter(delay=0.01)
        key = make_dedup_key("j", "x")

        try:
            tasks = [await engine.submit_dedup(key, lambda: counter.work("x")) for _ in range(4)]
            await engine.gather(tasks)
        finally:
            await engine.shutdown()

        assert engine.stats.tasks_submitted == 4
        assert engine.stats.tasks_completed == 4
