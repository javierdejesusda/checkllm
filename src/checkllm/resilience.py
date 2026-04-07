"""Rate limiting, circuit breaker, and retry patterns for checkllm judge backends.

Provides resilience wrappers that can be composed around any ``JudgeBackend``:

* **TokenBucketRateLimiter** -- classic token-bucket algorithm (async-safe).
* **PerProviderRateLimiter** -- separate buckets per provider name.
* **CircuitBreaker** -- opens after consecutive failures, auto-recovers.
* **ResilientJudge** -- combines rate limiting + circuit breaker + fallback.
* **RetryPolicy** / **with_retry** -- configurable retry with exponential backoff.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from checkllm.models import JudgeResponse

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Async-safe token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens replenished per second.
    burst:
        Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")

        self.rate = rate
        self.burst = burst

        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

        # Metrics
        self.total_acquired: int = 0
        self.total_waited_ms: float = 0.0

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Block until *tokens* are available, then consume them."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.burst:
            raise ValueError(
                f"requested {tokens} tokens exceeds burst size {self.burst}"
            )

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_acquired += tokens
                    return

                # Calculate how long until enough tokens accumulate
                deficit = tokens - self._tokens
                wait_seconds = deficit / self.rate

            start = time.monotonic()
            await asyncio.sleep(wait_seconds)
            self.total_waited_ms += (time.monotonic() - start) * 1000.0

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to consume *tokens* without blocking. Returns ``True`` on success."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")

        # Note: try_acquire is synchronous but we still need the logical refill.
        # Since it is non-blocking we skip the async lock and rely on the GIL for
        # basic safety.  For truly concurrent async usage, callers should prefer
        # ``acquire``.
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            self.total_acquired += tokens
            return True
        return False


class PerProviderRateLimiter:
    """Manages separate :class:`TokenBucketRateLimiter` instances per provider.

    A default limiter is used for providers that have not been explicitly
    registered via :meth:`add_provider`.

    Parameters
    ----------
    default_rate:
        Default tokens-per-second for unknown providers.
    default_burst:
        Default burst size for unknown providers.
    """

    def __init__(self, default_rate: float = 10.0, default_burst: int = 10) -> None:
        self._default_rate = default_rate
        self._default_burst = default_burst
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

    def add_provider(self, name: str, rate: float, burst: int) -> None:
        """Register a dedicated limiter for *name*."""
        self._limiters[name] = TokenBucketRateLimiter(rate=rate, burst=burst)

    def _get_limiter(self, provider: str) -> TokenBucketRateLimiter:
        if provider not in self._limiters:
            self._limiters[provider] = TokenBucketRateLimiter(
                rate=self._default_rate, burst=self._default_burst
            )
        return self._limiters[provider]

    async def acquire(self, provider: str, tokens: int = 1) -> None:
        """Acquire *tokens* from the limiter for *provider*."""
        await self._get_limiter(provider).acquire(tokens)

    def try_acquire(self, provider: str, tokens: int = 1) -> bool:
        """Non-blocking acquire for *provider*."""
        return self._get_limiter(provider).try_acquire(tokens)

    def get_limiter(self, provider: str) -> TokenBucketRateLimiter:
        """Return the limiter for *provider* (creates a default one if needed)."""
        return self._get_limiter(provider)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open.

    Attributes
    ----------
    time_until_retry:
        Seconds remaining before the circuit transitions to HALF_OPEN.
    """

    def __init__(self, time_until_retry: float) -> None:
        self.time_until_retry = max(0.0, time_until_retry)
        super().__init__(
            f"Circuit is OPEN. Retry after {self.time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker that wraps async callables.

    Parameters
    ----------
    failure_threshold:
        Consecutive failures required to open the circuit.
    recovery_timeout:
        Seconds the circuit stays open before moving to HALF_OPEN.
    half_open_max_calls:
        Maximum concurrent calls allowed while in HALF_OPEN state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

        # Counters
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float | None = None
        self.state_changes: list[tuple[CircuitState, CircuitState, float]] = []
        self._half_open_calls: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition from OPEN to HALF_OPEN)."""
        if self._state is CircuitState.OPEN and self.last_failure_time is not None:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
        return self._state

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        if old is not new_state:
            self._state = new_state
            self.state_changes.append((old, new_state, time.monotonic()))
            if new_state is CircuitState.HALF_OPEN:
                self._half_open_calls = 0

    async def call(self, coro: Awaitable[T]) -> T:
        """Execute *coro* through the circuit breaker.

        Raises :class:`CircuitOpenError` if the circuit is currently OPEN and
        the recovery timeout has not elapsed.
        """
        async with self._lock:
            current = self.state  # may auto-transition OPEN -> HALF_OPEN

            if current is CircuitState.OPEN:
                assert self.last_failure_time is not None
                remaining = self.recovery_timeout - (
                    time.monotonic() - self.last_failure_time
                )
                raise CircuitOpenError(remaining)

            if current is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(0.0)
                self._half_open_calls += 1

        # Execute outside the lock so we don't block other callers
        try:
            result = await coro
        except Exception:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.monotonic()
                if self._state is CircuitState.HALF_OPEN:
                    self._transition(CircuitState.OPEN)
                elif self.failure_count >= self.failure_threshold:
                    self._transition(CircuitState.OPEN)
            raise

        # Success
        async with self._lock:
            self.success_count += 1
            if self._state is CircuitState.HALF_OPEN:
                # Successful probe -- close the circuit
                self.failure_count = 0
                self._transition(CircuitState.CLOSED)
        return result

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self.failure_count = 0
        self._half_open_calls = 0


# ---------------------------------------------------------------------------
# Resilient Judge
# ---------------------------------------------------------------------------

class ResilientJudge:
    """Wraps a :class:`JudgeBackend` with rate limiting, circuit breaking,
    timeout, and fallback capabilities.

    Implements the ``JudgeBackend`` protocol so it can be used as a drop-in
    replacement anywhere a judge is expected.

    Parameters
    ----------
    judge:
        The primary judge backend.
    rate_limiter:
        Optional rate limiter applied before each call.
    circuit_breaker:
        Optional circuit breaker wrapping each call.
    fallback:
        Optional fallback judge used when the primary fails or circuit is open.
    timeout:
        Optional per-call timeout in seconds.
    """

    def __init__(
        self,
        judge: Any,  # JudgeBackend protocol
        rate_limiter: TokenBucketRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        fallback: Any | None = None,  # JudgeBackend protocol
        timeout: float | None = None,
    ) -> None:
        self._judge = judge
        self._rate_limiter = rate_limiter
        self._circuit_breaker = circuit_breaker
        self._fallback = fallback
        self._timeout = timeout

        # Metrics
        self.primary_calls: int = 0
        self.fallback_calls: int = 0
        self.circuit_opens: int = 0

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        """Evaluate with resilience: rate-limit -> circuit-break -> timeout -> fallback."""

        # Rate limiting
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        # If circuit is open and we have a fallback, go there directly
        if self._circuit_breaker is not None:
            current_state = self._circuit_breaker.state
            if current_state is CircuitState.OPEN and self._fallback is not None:
                self.circuit_opens += 1
                return await self._call_fallback(prompt, system_prompt)

        # Try primary
        try:
            result = await self._call_primary(prompt, system_prompt)
            self.primary_calls += 1
            return result
        except CircuitOpenError:
            if self._fallback is not None:
                self.circuit_opens += 1
                return await self._call_fallback(prompt, system_prompt)
            raise
        except Exception:
            if self._fallback is not None:
                return await self._call_fallback(prompt, system_prompt)
            raise

    async def _call_primary(
        self, prompt: str, system_prompt: str | None
    ) -> JudgeResponse:
        """Call the primary judge with optional circuit breaker and timeout."""
        coro = self._judge.evaluate(prompt, system_prompt)

        if self._timeout is not None:
            coro = asyncio.wait_for(coro, timeout=self._timeout)

        if self._circuit_breaker is not None:
            return await self._circuit_breaker.call(coro)

        return await coro

    async def _call_fallback(
        self, prompt: str, system_prompt: str | None
    ) -> JudgeResponse:
        """Call the fallback judge."""
        self.fallback_calls += 1
        return await self._fallback.evaluate(prompt, system_prompt)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Configuration for retry with exponential backoff.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts (not counting the initial try).
    base_delay:
        Base delay in seconds for the first retry.
    max_delay:
        Upper bound on the delay in seconds.
    exponential_base:
        Multiplier applied to each successive retry delay.
    jitter:
        If ``True``, add random jitter (0 -- 100% of computed delay).
    retry_on:
        Exception types that trigger a retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

    def delay_for(self, attempt: int) -> float:
        """Compute the delay in seconds for the given *attempt* (0-indexed).

        Returns the raw exponential delay (capped at ``max_delay``), **without**
        jitter applied -- jitter is added at call time so that
        ``delay_for`` remains deterministic for testing.
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


async def with_retry(
    coro_fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
) -> T:
    """Execute *coro_fn* with retries according to *policy*.

    Parameters
    ----------
    coro_fn:
        A zero-argument callable that returns a fresh awaitable each time it
        is invoked (e.g. ``lambda: judge.evaluate(prompt)``).
    policy:
        Retry configuration.  Uses defaults if ``None``.

    Raises the last exception if all retries are exhausted.
    """
    if policy is None:
        policy = RetryPolicy()

    last_exc: Exception | None = None
    for attempt in range(1 + policy.max_retries):
        try:
            return await coro_fn()
        except policy.retry_on as exc:
            last_exc = exc
            if attempt < policy.max_retries:
                delay = policy.delay_for(attempt)
                if policy.jitter:
                    delay *= random.uniform(0.0, 1.0)
                await asyncio.sleep(delay)

    # All retries exhausted
    assert last_exc is not None
    raise last_exc
