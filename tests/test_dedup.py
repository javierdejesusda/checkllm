"""Tests for InFlightDeduplicator.

Verifies that concurrent calls with the same key coalesce onto a single
invocation of the supplied factory, while distinct keys remain independent.
"""

from __future__ import annotations

import asyncio

import pytest

from checkllm.dedup import InFlightDeduplicator, make_dedup_key


class _Counter:
    """Mock judge-like object that counts invocations."""

    def __init__(self, delay: float = 0.02) -> None:
        self.calls = 0
        self.delay = delay

    async def evaluate(self, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return f"result:{prompt}"


class TestMakeDedupKey:
    def test_same_inputs_produce_same_key(self) -> None:
        k1 = make_dedup_key("openai", "hi", model="gpt-4o", temperature=0.0)
        k2 = make_dedup_key("openai", "hi", model="gpt-4o", temperature=0.0)
        assert k1 == k2

    def test_different_prompts_produce_different_keys(self) -> None:
        k1 = make_dedup_key("openai", "a", model="gpt-4o")
        k2 = make_dedup_key("openai", "b", model="gpt-4o")
        assert k1 != k2

    def test_different_models_produce_different_keys(self) -> None:
        k1 = make_dedup_key("openai", "a", model="gpt-4o")
        k2 = make_dedup_key("openai", "a", model="gpt-4o-mini")
        assert k1 != k2

    def test_key_is_stable_across_extra_ordering(self) -> None:
        k1 = make_dedup_key("o", "x", extra_a=1, extra_b=2)
        k2 = make_dedup_key("o", "x", extra_b=2, extra_a=1)
        assert k1 == k2


class TestInFlightDeduplicator:
    @pytest.mark.asyncio
    async def test_single_call_not_deduplicated(self) -> None:
        dedup = InFlightDeduplicator()
        counter = _Counter()
        key = make_dedup_key("j", "hi")

        result = await dedup.run(key, lambda: counter.evaluate("hi"))

        assert result == "result:hi"
        assert counter.calls == 1
        assert dedup.stats() == {"hits": 0, "misses": 1, "inflight": 0}

    @pytest.mark.asyncio
    async def test_concurrent_same_key_coalesces(self) -> None:
        dedup = InFlightDeduplicator()
        counter = _Counter(delay=0.05)
        key = make_dedup_key("j", "hi")

        results = await asyncio.gather(
            *[dedup.run(key, lambda: counter.evaluate("hi")) for _ in range(10)]
        )

        assert all(r == "result:hi" for r in results)
        assert counter.calls == 1, "factory should run exactly once"
        assert dedup.hits == 9
        assert dedup.misses == 1

    @pytest.mark.asyncio
    async def test_different_keys_run_independently(self) -> None:
        dedup = InFlightDeduplicator()
        counter = _Counter(delay=0.02)

        keys = [make_dedup_key("j", p) for p in ("a", "b", "c")]
        results = await asyncio.gather(
            *[dedup.run(k, lambda p=p: counter.evaluate(p)) for k, p in zip(keys, ("a", "b", "c"))]
        )

        assert set(results) == {"result:a", "result:b", "result:c"}
        assert counter.calls == 3
        assert dedup.misses == 3
        assert dedup.hits == 0

    @pytest.mark.asyncio
    async def test_cleared_after_completion(self) -> None:
        dedup = InFlightDeduplicator()
        counter = _Counter()
        key = make_dedup_key("j", "x")

        await dedup.run(key, lambda: counter.evaluate("x"))
        # The second call happens after the first resolved so it should
        # count as a fresh miss (not a coalesced hit).
        await dedup.run(key, lambda: counter.evaluate("x"))

        assert counter.calls == 2
        assert dedup.misses == 2
        assert dedup.hits == 0

    @pytest.mark.asyncio
    async def test_exception_propagates_to_all_waiters(self) -> None:
        dedup = InFlightDeduplicator()

        class _Boom(RuntimeError):
            pass

        attempts = {"n": 0}

        async def boom() -> str:
            attempts["n"] += 1
            await asyncio.sleep(0.01)
            raise _Boom("nope")

        key = make_dedup_key("j", "x")
        tasks = [asyncio.create_task(dedup.run(key, boom)) for _ in range(4)]
        for t in tasks:
            with pytest.raises(_Boom):
                await t

        assert attempts["n"] == 1
        # Key should be cleared so a retry starts fresh.
        assert len(dedup._inflight) == 0

    @pytest.mark.asyncio
    async def test_reset_stats(self) -> None:
        dedup = InFlightDeduplicator()
        key = make_dedup_key("j", "x")
        await dedup.run(key, lambda: asyncio.sleep(0, result="done"))

        assert dedup.stats()["misses"] == 1
        dedup.reset_stats()
        assert dedup.stats() == {"hits": 0, "misses": 0, "inflight": 0}
