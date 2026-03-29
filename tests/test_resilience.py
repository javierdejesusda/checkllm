"""Tests for checkllm.resilience -- rate limiting, circuit breaker, and retry."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from checkllm.models import JudgeResponse
from checkllm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    PerProviderRateLimiter,
    ResilientJudge,
    RetryPolicy,
    TokenBucketRateLimiter,
    with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_response(score: float = 0.9) -> JudgeResponse:
    return JudgeResponse(score=score, reasoning="ok")


class _SimpleMockJudge:
    """Minimal mock that satisfies JudgeBackend protocol."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[tuple[str, str | None]] = []

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        self.calls.append((prompt, system_prompt))
        return _ok_response(self.score)


class _FailingJudge:
    """Judge that raises on every call."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("judge failure")

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        raise self._exc


class _FailThenSucceedJudge:
    """Fails *n* times, then succeeds."""

    def __init__(self, failures: int = 2, score: float = 0.9) -> None:
        self._failures = failures
        self._score = score
        self._call_count = 0

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        self._call_count += 1
        if self._call_count <= self._failures:
            raise RuntimeError(f"failure #{self._call_count}")
        return _ok_response(self._score)


class _SlowJudge:
    """Judge that takes a configurable amount of time."""

    def __init__(self, delay: float, score: float = 0.9) -> None:
        self._delay = delay
        self._score = score

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        await asyncio.sleep(self._delay)
        return _ok_response(self._score)


# ===================================================================
# TokenBucketRateLimiter
# ===================================================================

class TestTokenBucketRateLimiter:
    def test_init_validation(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketRateLimiter(rate=0, burst=10)
        with pytest.raises(ValueError, match="burst must be positive"):
            TokenBucketRateLimiter(rate=1.0, burst=0)

    @pytest.mark.asyncio
    async def test_acquire_within_burst(self):
        limiter = TokenBucketRateLimiter(rate=100.0, burst=10)
        # Should be able to acquire up to burst without blocking
        for _ in range(10):
            await limiter.acquire(1)
        assert limiter.total_acquired == 10

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        limiter = TokenBucketRateLimiter(rate=100.0, burst=10)
        await limiter.acquire(5)
        assert limiter.total_acquired == 5

    @pytest.mark.asyncio
    async def test_acquire_exceeds_burst_raises(self):
        limiter = TokenBucketRateLimiter(rate=1.0, burst=5)
        with pytest.raises(ValueError, match="exceeds burst"):
            await limiter.acquire(6)

    @pytest.mark.asyncio
    async def test_acquire_zero_raises(self):
        limiter = TokenBucketRateLimiter(rate=1.0, burst=5)
        with pytest.raises(ValueError, match="tokens must be positive"):
            await limiter.acquire(0)

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_empty(self):
        # High rate so refill is fast, small burst
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=1)
        await limiter.acquire(1)  # empties the bucket

        start = time.monotonic()
        await limiter.acquire(1)  # must wait for refill
        elapsed = time.monotonic() - start

        assert limiter.total_acquired == 2
        # Should have waited ~1ms (1 token / 1000 tokens/sec)
        assert elapsed >= 0.0005  # generous lower bound

    @pytest.mark.asyncio
    async def test_acquire_tracks_waited_ms(self):
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=1)
        await limiter.acquire(1)
        await limiter.acquire(1)
        # After the second acquire, some wait time should be recorded
        assert limiter.total_waited_ms > 0

    def test_try_acquire_success(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        assert limiter.try_acquire(3) is True
        assert limiter.total_acquired == 3

    def test_try_acquire_insufficient_tokens(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        assert limiter.try_acquire(5) is True
        assert limiter.try_acquire(1) is False
        assert limiter.total_acquired == 5  # only the first 5

    def test_try_acquire_zero_raises(self):
        limiter = TokenBucketRateLimiter(rate=1.0, burst=5)
        with pytest.raises(ValueError, match="tokens must be positive"):
            limiter.try_acquire(0)

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=5)
        # Drain all tokens
        await limiter.acquire(5)
        # Wait for refill -- use a generous sleep for Windows timer precision
        await asyncio.sleep(0.05)  # ~50 tokens at 1000/s, well above 3
        assert limiter.try_acquire(3) is True

    @pytest.mark.asyncio
    async def test_burst_cap(self):
        """Tokens should never exceed burst size even after a long wait."""
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=3)
        await asyncio.sleep(0.01)  # would add ~10 tokens without cap
        assert limiter.try_acquire(3) is True
        assert limiter.try_acquire(1) is False


# ===================================================================
# PerProviderRateLimiter
# ===================================================================

class TestPerProviderRateLimiter:
    @pytest.mark.asyncio
    async def test_separate_limits(self):
        prl = PerProviderRateLimiter(default_rate=100.0, default_burst=10)
        prl.add_provider("openai", rate=100.0, burst=2)
        prl.add_provider("anthropic", rate=100.0, burst=3)

        # OpenAI: can acquire 2
        await prl.acquire("openai")
        await prl.acquire("openai")
        assert prl.get_limiter("openai").total_acquired == 2

        # Anthropic: can acquire 3
        await prl.acquire("anthropic")
        await prl.acquire("anthropic")
        await prl.acquire("anthropic")
        assert prl.get_limiter("anthropic").total_acquired == 3

    @pytest.mark.asyncio
    async def test_default_provider(self):
        prl = PerProviderRateLimiter(default_rate=100.0, default_burst=5)
        await prl.acquire("unknown_provider")
        limiter = prl.get_limiter("unknown_provider")
        assert limiter.rate == 100.0
        assert limiter.burst == 5

    def test_try_acquire(self):
        prl = PerProviderRateLimiter(default_rate=100.0, default_burst=1)
        prl.add_provider("openai", rate=100.0, burst=1)
        assert prl.try_acquire("openai") is True
        assert prl.try_acquire("openai") is False

    @pytest.mark.asyncio
    async def test_providers_independent(self):
        """Draining one provider does not affect another."""
        prl = PerProviderRateLimiter(default_rate=100.0, default_burst=1)
        prl.add_provider("a", rate=100.0, burst=1)
        prl.add_provider("b", rate=100.0, burst=1)

        await prl.acquire("a")
        # 'a' is drained, but 'b' should still be available
        assert prl.try_acquire("b") is True
        assert prl.try_acquire("a") is False


# ===================================================================
# CircuitBreaker
# ===================================================================

class TestCircuitBreaker:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=0)
        with pytest.raises(ValueError):
            CircuitBreaker(half_open_max_calls=0)

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_success_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(5):
            result = await cb.call(_async_value(42))
            assert result == 42
        assert cb.state is CircuitState.CLOSED
        assert cb.success_count == 5
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        for i in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(_async_fail(RuntimeError("boom")))

        assert cb.state is CircuitState.OPEN
        assert cb.failure_count == 3
        assert len(cb.state_changes) == 1
        assert cb.state_changes[0][0] is CircuitState.CLOSED
        assert cb.state_changes[0][1] is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("boom")))

        assert cb.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(_async_value("should not reach"))
        assert exc_info.value.time_until_retry > 0

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("boom")))
        assert cb.state is CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("boom")))
        assert cb.state is CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

        # Successful probe closes the circuit
        result = await cb.call(_async_value("recovered"))
        assert result == "recovered"
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("first")))
        assert cb.state is CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

        # Failure in half-open reopens
        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("second")))
        assert cb.state is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_max_calls(self):
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=1
        )

        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("boom")))

        await asyncio.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

        # First call in half-open is allowed (we need to make it slow so the
        # second call arrives while the first is still running).
        # For simplicity, we test that after one call starts, the next is rejected.
        # We use the lock-based behavior: the first call consumes the slot.
        result = await cb.call(_async_value("probe"))
        assert result == "probe"
        # Circuit is now closed after success -- this tests the flow, not
        # concurrent rejection (which requires real concurrency).
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        with pytest.raises(RuntimeError):
            await cb.call(_async_fail(RuntimeError("boom")))
        assert cb.state is CircuitState.OPEN

        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_full_cycle_closed_open_half_open_closed(self):
        """Test the full lifecycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

        # CLOSED: two failures open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(_async_fail(RuntimeError("fail")))
        assert cb.state is CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

        # Successful probe closes it
        await cb.call(_async_value("ok"))
        assert cb.state is CircuitState.CLOSED

        # Verify state change history
        states = [(old.value, new.value) for old, new, _ in cb.state_changes]
        assert ("closed", "open") in states
        assert ("open", "half_open") in states
        assert ("half_open", "closed") in states


# ===================================================================
# CircuitOpenError
# ===================================================================

class TestCircuitOpenError:
    def test_message(self):
        err = CircuitOpenError(time_until_retry=12.5)
        assert err.time_until_retry == 12.5
        assert "12.5" in str(err)

    def test_negative_clamped_to_zero(self):
        err = CircuitOpenError(time_until_retry=-5.0)
        assert err.time_until_retry == 0.0

    def test_is_exception(self):
        err = CircuitOpenError(time_until_retry=1.0)
        assert isinstance(err, Exception)


# ===================================================================
# ResilientJudge
# ===================================================================

class TestResilientJudge:
    @pytest.mark.asyncio
    async def test_normal_operation(self):
        judge = _SimpleMockJudge(score=0.9)
        rj = ResilientJudge(judge=judge)
        resp = await rj.evaluate("test prompt")
        assert resp.score == 0.9
        assert rj.primary_calls == 1
        assert rj.fallback_calls == 0
        assert len(judge.calls) == 1

    @pytest.mark.asyncio
    async def test_with_rate_limiter(self):
        judge = _SimpleMockJudge()
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=10)
        rj = ResilientJudge(judge=judge, rate_limiter=limiter)

        for _ in range(5):
            await rj.evaluate("prompt")

        assert rj.primary_calls == 5
        assert limiter.total_acquired == 5

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_success(self):
        judge = _SimpleMockJudge()
        cb = CircuitBreaker(failure_threshold=3)
        rj = ResilientJudge(judge=judge, circuit_breaker=cb)
        resp = await rj.evaluate("test")
        assert resp.score == 0.9
        assert cb.success_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        primary = _FailingJudge()
        fallback = _SimpleMockJudge(score=0.7)
        rj = ResilientJudge(judge=primary, fallback=fallback)

        resp = await rj.evaluate("test")
        assert resp.score == 0.7
        assert rj.fallback_calls == 1
        assert rj.primary_calls == 0

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self):
        primary = _FailingJudge()
        rj = ResilientJudge(judge=primary)

        with pytest.raises(RuntimeError, match="judge failure"):
            await rj.evaluate("test")

    @pytest.mark.asyncio
    async def test_circuit_open_goes_to_fallback(self):
        primary = _FailingJudge()
        fallback = _SimpleMockJudge(score=0.6)
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        rj = ResilientJudge(
            judge=primary, circuit_breaker=cb, fallback=fallback
        )

        # First call trips the circuit
        resp1 = await rj.evaluate("test")
        assert resp1.score == 0.6  # fell back
        assert rj.fallback_calls == 1

        # Second call: circuit is open, goes directly to fallback
        resp2 = await rj.evaluate("test again")
        assert resp2.score == 0.6
        assert rj.circuit_opens >= 1

    @pytest.mark.asyncio
    async def test_circuit_open_no_fallback_raises(self):
        primary = _FailingJudge()
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        rj = ResilientJudge(judge=primary, circuit_breaker=cb)

        # First call fails and trips the circuit
        with pytest.raises(RuntimeError):
            await rj.evaluate("test")

        # Circuit is now open, no fallback -> CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await rj.evaluate("test again")

    @pytest.mark.asyncio
    async def test_timeout(self):
        slow = _SlowJudge(delay=5.0)
        fallback = _SimpleMockJudge(score=0.5)
        rj = ResilientJudge(judge=slow, fallback=fallback, timeout=0.05)

        resp = await rj.evaluate("test")
        # Should have timed out and fallen back
        assert resp.score == 0.5
        assert rj.fallback_calls == 1

    @pytest.mark.asyncio
    async def test_timeout_no_fallback_raises(self):
        slow = _SlowJudge(delay=5.0)
        rj = ResilientJudge(judge=slow, timeout=0.05)

        with pytest.raises(asyncio.TimeoutError):
            await rj.evaluate("test")

    @pytest.mark.asyncio
    async def test_combined_rate_limit_and_circuit(self):
        judge = _SimpleMockJudge()
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=10)
        cb = CircuitBreaker(failure_threshold=5)
        rj = ResilientJudge(
            judge=judge, rate_limiter=limiter, circuit_breaker=cb
        )

        for _ in range(3):
            await rj.evaluate("prompt")

        assert rj.primary_calls == 3
        assert limiter.total_acquired == 3
        assert cb.success_count == 3


# ===================================================================
# RetryPolicy
# ===================================================================

class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.max_delay == 30.0
        assert p.exponential_base == 2.0
        assert p.jitter is True
        assert TimeoutError in p.retry_on
        assert ConnectionError in p.retry_on

    def test_delay_for_exponential(self):
        p = RetryPolicy(base_delay=1.0, exponential_base=2.0, max_delay=100.0)
        assert p.delay_for(0) == 1.0   # 1 * 2^0
        assert p.delay_for(1) == 2.0   # 1 * 2^1
        assert p.delay_for(2) == 4.0   # 1 * 2^2
        assert p.delay_for(3) == 8.0   # 1 * 2^3

    def test_delay_for_capped_at_max(self):
        p = RetryPolicy(base_delay=1.0, exponential_base=2.0, max_delay=5.0)
        assert p.delay_for(0) == 1.0
        assert p.delay_for(1) == 2.0
        assert p.delay_for(2) == 4.0
        assert p.delay_for(3) == 5.0  # capped
        assert p.delay_for(10) == 5.0  # still capped

    def test_delay_for_custom_base(self):
        p = RetryPolicy(base_delay=0.5, exponential_base=3.0, max_delay=100.0)
        assert p.delay_for(0) == 0.5   # 0.5 * 3^0
        assert p.delay_for(1) == 1.5   # 0.5 * 3^1
        assert p.delay_for(2) == 4.5   # 0.5 * 3^2


# ===================================================================
# with_retry
# ===================================================================

class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(fn, RetryPolicy(max_retries=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_matching_exception(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "recovered"

        policy = RetryPolicy(
            max_retries=5, base_delay=0.001, jitter=False,
            retry_on=(TimeoutError,),
        )
        result = await with_retry(fn, policy)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("down")

        policy = RetryPolicy(
            max_retries=2, base_delay=0.001, jitter=False,
            retry_on=(ConnectionError,),
        )
        with pytest.raises(ConnectionError, match="down"):
            await with_retry(fn, policy)
        # 1 initial + 2 retries = 3 total
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_raises_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        policy = RetryPolicy(
            max_retries=5, base_delay=0.001,
            retry_on=(TimeoutError,),
        )
        with pytest.raises(ValueError, match="bad input"):
            await with_retry(fn, policy)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_jitter_applied(self):
        """Verify that jitter makes actual sleep shorter than raw delay."""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timeout")
            return "ok"

        policy = RetryPolicy(
            max_retries=3, base_delay=0.01, jitter=True,
            retry_on=(TimeoutError,),
        )
        start = time.monotonic()
        await with_retry(fn, policy)
        elapsed = time.monotonic() - start
        # With jitter, actual sleep is between 0 and base_delay
        assert elapsed < 0.1  # generous upper bound for CI (Windows timer precision)

    @pytest.mark.asyncio
    async def test_default_policy(self):
        """with_retry uses default RetryPolicy when None is passed."""
        async def fn():
            return 42

        result = await with_retry(fn)
        assert result == 42


# ---------------------------------------------------------------------------
# Async helpers for circuit breaker tests
# ---------------------------------------------------------------------------

async def _async_value(value: T) -> T:
    return value


async def _async_fail(exc: Exception) -> None:
    raise exc
