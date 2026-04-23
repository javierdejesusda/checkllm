"""Per-provider RPM/TPM rate limiting with 429-aware retry.

This module complements :mod:`checkllm.resilience` (which provides a single
token-bucket plus circuit breaker) with a provider-aware dual-bucket limiter
that polices both *requests per minute* (RPM) and *tokens per minute* (TPM),
and a 429-aware retry helper that honors ``Retry-After`` /
``x-ratelimit-reset-*`` response headers.

Typical wiring is:

    1.  An engine (e.g. :class:`checkllm.engines.AsyncEngine`) receives a
        coroutine that will call a provider API.
    2.  Before executing the coroutine, the engine calls
        :meth:`ProviderRateLimiter.acquire` with an estimated token count.
        This blocks until both the RPM and TPM buckets have sufficient
        capacity.
    3.  If the underlying call raises a 429 (or compatible 5xx), the engine
        routes the retry through :func:`retry_with_backoff`, which respects
        ``Retry-After`` hints when present.
    4.  Post-call, :meth:`ProviderRateLimiter.release_actual` is called with
        the *actual* tokens consumed so the TPM bucket can reconcile its
        estimate against reality.

The default per-provider limits are intentionally conservative. They target
tier-1 / free-tier accounts so users are not surprised by upstream 429s.
Override them via :class:`checkllm.config.CheckllmConfig.rate_limits`.

Default rate table (RPM / TPM per minute):

+---------------+------+------------+
| Provider      |  RPM |        TPM |
+===============+======+============+
| openai        |  500 |     30_000 |
| anthropic     |  500 |     40_000 |
| gemini        |  360 |    120_000 |
| azure         |  240 |     30_000 |
| bedrock       |  200 |     60_000 |
| cohere        |  100 |     20_000 |
| mistral       |  120 |     20_000 |
| groq          |  300 |     15_000 |
| together      |  600 |    100_000 |
| fireworks     |  600 |    100_000 |
| perplexity    |   60 |     10_000 |
| deepseek      |  500 |     60_000 |
| openrouter    |  200 |     40_000 |
| xai           |   60 |     10_000 |
| vllm          | 1000 |    500_000 |
| ollama        | 1000 |  1_000_000 |
| litellm       |  200 |     20_000 |
| custom        |  100 |     10_000 |
+---------------+------+------------+
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, TypeVar

logger = logging.getLogger("checkllm.rate_limit")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimit:
    """Static rate-limit configuration for a provider.

    Attributes:
        rpm: Maximum requests per minute.
        tpm: Maximum tokens per minute.
    """

    rpm: int
    tpm: int


# Conservative per-provider defaults — see module docstring.
DEFAULT_LIMITS: dict[str, RateLimit] = {
    "openai": RateLimit(rpm=500, tpm=30_000),
    "anthropic": RateLimit(rpm=500, tpm=40_000),
    "gemini": RateLimit(rpm=360, tpm=120_000),
    "azure": RateLimit(rpm=240, tpm=30_000),
    "bedrock": RateLimit(rpm=200, tpm=60_000),
    "cohere": RateLimit(rpm=100, tpm=20_000),
    "mistral": RateLimit(rpm=120, tpm=20_000),
    "groq": RateLimit(rpm=300, tpm=15_000),
    "together": RateLimit(rpm=600, tpm=100_000),
    "fireworks": RateLimit(rpm=600, tpm=100_000),
    "perplexity": RateLimit(rpm=60, tpm=10_000),
    "deepseek": RateLimit(rpm=500, tpm=60_000),
    "openrouter": RateLimit(rpm=200, tpm=40_000),
    "xai": RateLimit(rpm=60, tpm=10_000),
    "vllm": RateLimit(rpm=1000, tpm=500_000),
    "ollama": RateLimit(rpm=1000, tpm=1_000_000),
    "litellm": RateLimit(rpm=200, tpm=20_000),
    "custom": RateLimit(rpm=100, tpm=10_000),
}

DEFAULT_FALLBACK = RateLimit(rpm=120, tpm=20_000)


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """Continuously-refilled token bucket, async-safe.

    Unlike a fixed-window counter, this bucket refills smoothly over time
    (``capacity`` tokens per ``refill_period`` seconds). Calls to
    :meth:`acquire` block until enough tokens are available.

    Parameters:
        capacity: Maximum tokens the bucket can hold.
        refill_period: Seconds over which ``capacity`` tokens are refilled.
            For an "RPM" bucket use ``refill_period=60.0``.
        initial: Initial token count (defaults to ``capacity``).

    The bucket is monotonic: time never goes backwards.
    """

    def __init__(
        self,
        capacity: float,
        refill_period: float = 60.0,
        *,
        initial: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_period <= 0:
            raise ValueError("refill_period must be positive")

        self.capacity = float(capacity)
        self.refill_period = float(refill_period)
        self.refill_rate = self.capacity / self.refill_period  # tokens per second

        self._tokens: float = float(capacity if initial is None else initial)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

        # Metrics
        self.total_acquired: float = 0.0
        self.total_wait_seconds: float = 0.0

    # -- internals ----------------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

    @property
    def available(self) -> float:
        """Non-blocking peek at the current token count (refills first)."""
        self._refill()
        return self._tokens

    # -- public API ---------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* are available, then consume them.

        Args:
            tokens: Amount to consume. Must be positive and ``<= capacity``.

        Raises:
            ValueError: If ``tokens`` is non-positive or exceeds capacity.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.capacity:
            raise ValueError(f"requested {tokens} tokens exceeds bucket capacity {self.capacity}")

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_acquired += tokens
                    return
                deficit = tokens - self._tokens
                wait_seconds = deficit / self.refill_rate

            start = time.monotonic()
            await asyncio.sleep(wait_seconds)
            self.total_wait_seconds += time.monotonic() - start

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking variant of :meth:`acquire`.

        Returns:
            ``True`` if ``tokens`` were consumed, ``False`` otherwise.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            self.total_acquired += tokens
            return True
        return False

    def refund(self, tokens: float) -> None:
        """Return ``tokens`` to the bucket, capped at ``capacity``.

        Useful when an estimate was too pessimistic.
        """
        if tokens <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + tokens)


# ---------------------------------------------------------------------------
# ProviderRateLimiter
# ---------------------------------------------------------------------------


@dataclass
class _ProviderState:
    """Per-provider pair of RPM + TPM buckets."""

    limit: RateLimit
    rpm_bucket: TokenBucket
    tpm_bucket: TokenBucket
    # Reconciliation ledger: how many tokens were pre-charged via estimates
    # that have not yet been reconciled by a release_actual() call.
    outstanding: float = 0.0
    # Metrics
    rate_429_hits: int = 0


class ProviderRateLimiter:
    """Registry of per-provider :class:`TokenBucket` pairs.

    Each provider name maps to an RPM bucket (1 token per request) and a
    TPM bucket (1 token per estimated model token). Both buckets must
    admit a call before it proceeds.

    Parameters:
        limits: Mapping from provider name to :class:`RateLimit`. Defaults
            to :data:`DEFAULT_LIMITS` with unknown providers falling back to
            :data:`DEFAULT_FALLBACK`.
        default_limit: Fallback for providers not in ``limits``.
    """

    def __init__(
        self,
        limits: Mapping[str, RateLimit] | None = None,
        *,
        default_limit: RateLimit = DEFAULT_FALLBACK,
    ) -> None:
        self._limits: dict[str, RateLimit] = dict(DEFAULT_LIMITS)
        if limits:
            self._limits.update(limits)
        self._default_limit = default_limit
        self._providers: dict[str, _ProviderState] = {}
        self._registry_lock = asyncio.Lock()

    # -- registry helpers ---------------------------------------------------

    def _ensure_provider(self, provider: str) -> _ProviderState:
        state = self._providers.get(provider)
        if state is not None:
            return state
        limit = self._limits.get(provider, self._default_limit)
        rpm = TokenBucket(capacity=limit.rpm, refill_period=60.0)
        tpm = TokenBucket(capacity=limit.tpm, refill_period=60.0)
        state = _ProviderState(limit=limit, rpm_bucket=rpm, tpm_bucket=tpm)
        self._providers[provider] = state
        return state

    def configure(self, provider: str, limit: RateLimit) -> None:
        """Register (or replace) the rate limit for *provider*.

        Replaces any in-flight bucket. Call before traffic starts.
        """
        self._limits[provider] = limit
        rpm = TokenBucket(capacity=limit.rpm, refill_period=60.0)
        tpm = TokenBucket(capacity=limit.tpm, refill_period=60.0)
        self._providers[provider] = _ProviderState(limit=limit, rpm_bucket=rpm, tpm_bucket=tpm)

    def get_state(self, provider: str) -> _ProviderState:
        """Return the internal state for ``provider`` (mostly for tests)."""
        return self._ensure_provider(provider)

    # -- public API ---------------------------------------------------------

    async def acquire(self, provider: str, est_tokens: int = 1) -> None:
        """Wait for RPM + TPM capacity on *provider* to admit a call.

        Args:
            provider: Provider name (e.g. ``"openai"``, ``"anthropic"``).
            est_tokens: Estimated token cost of the upcoming request. Must be
                positive. If this exceeds the provider's TPM capacity the
                request is still admitted (after draining the bucket) — we
                warn rather than raise to keep large prompts working.
        """
        if est_tokens <= 0:
            est_tokens = 1

        state = self._ensure_provider(provider)

        # Clamp token request to bucket capacity to avoid ValueError — the
        # user can't make a request that inherently exceeds the per-minute
        # budget smaller by retrying, so we drain the bucket and let the
        # provider return the real 429 if it's overcapacity.
        token_charge = min(float(est_tokens), state.tpm_bucket.capacity)
        if token_charge < est_tokens:
            logger.warning(
                "rate_limit: %s est_tokens=%d exceeds TPM capacity=%d; clamping",
                provider,
                est_tokens,
                int(state.tpm_bucket.capacity),
            )

        await state.rpm_bucket.acquire(1.0)
        await state.tpm_bucket.acquire(token_charge)
        state.outstanding += token_charge

    def release_actual(self, provider: str, actual_tokens: int) -> None:
        """Reconcile an estimate against the tokens actually consumed.

        Call this after a successful API response returns real usage.
        If ``actual_tokens`` is less than the estimate, the difference is
        refunded to the TPM bucket. If it is greater, the extra is
        additionally charged (non-blocking; may drive the bucket below
        zero, which just means the next acquire waits longer).
        """
        state = self._providers.get(provider)
        if state is None:
            return
        actual = float(max(0, actual_tokens))
        # Pair this release with the most recent acquisition where possible;
        # if multiple calls are in flight we just reconcile against the pool.
        estimated = state.outstanding if state.outstanding > 0 else actual
        state.outstanding = max(0.0, state.outstanding - min(estimated, actual))

        delta = estimated - actual
        if delta > 0:
            state.tpm_bucket.refund(delta)
        elif delta < 0:
            # Under-estimated: deduct extra tokens directly.
            extra = -delta
            state.tpm_bucket._tokens = max(
                -state.tpm_bucket.capacity,  # soft floor
                state.tpm_bucket._tokens - extra,
            )

    def record_429(self, provider: str) -> None:
        """Bookkeeping: note that *provider* responded with a 429."""
        state = self._ensure_provider(provider)
        state.rate_429_hits += 1


# Process-wide default instance (callers may replace this).
_DEFAULT_LIMITER: ProviderRateLimiter | None = None


def get_default_limiter() -> ProviderRateLimiter:
    """Return the process-wide :class:`ProviderRateLimiter` singleton."""
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        _DEFAULT_LIMITER = ProviderRateLimiter()
    return _DEFAULT_LIMITER


def set_default_limiter(limiter: ProviderRateLimiter | None) -> None:
    """Replace (or clear) the process-wide limiter. Useful for tests."""
    global _DEFAULT_LIMITER
    _DEFAULT_LIMITER = limiter


# ---------------------------------------------------------------------------
# 429 detection + Retry-After parsing
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Canonical 429 raised by or wrapped around provider clients.

    Attributes:
        retry_after: Seconds to wait before retrying, or ``None``.
        provider: Optional provider name for bookkeeping.
    """

    def __init__(
        self,
        message: str = "rate limited",
        *,
        retry_after: float | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider


def parse_retry_after(value: str | float | int | None) -> float | None:
    """Parse a ``Retry-After`` header value into seconds.

    Supports both the integer-seconds form and the HTTP-date form.
    Returns ``None`` when the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    stripped = str(value).strip()
    if not stripped:
        return None
    # Integer seconds
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    # HTTP-date
    try:
        dt = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    now = time.time()
    delta = dt.timestamp() - now
    return max(0.0, delta)


def parse_reset_header(value: str | float | int | None) -> float | None:
    """Parse ``x-ratelimit-reset-*`` header values.

    Accepts either "seconds until reset" (common OpenAI form, may be a
    string like ``"3s"`` or ``"1m30s"``) or a unix timestamp. Returns
    seconds to wait, or ``None`` on failure.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: if it's a big number, treat as epoch; otherwise delta.
        v = float(value)
        if v > 10_000_000_000:  # > year 2286 if epoch
            v = v / 1000.0  # ms
        if v > 1_700_000_000:  # plausible epoch seconds
            return max(0.0, v - time.time())
        return max(0.0, v)
    stripped = str(value).strip().lower()
    if not stripped:
        return None
    # Handle compound forms like "1m30s", "500ms", "2.5s"
    total = 0.0
    num_buf = ""
    unit_buf = ""
    matched = False
    for ch in stripped + " ":
        if ch.isdigit() or ch == "." or ch == "-":
            if unit_buf and num_buf:
                total += _convert_unit(num_buf, unit_buf)
                num_buf = ""
                unit_buf = ""
                matched = True
            num_buf += ch
        elif ch.isalpha():
            unit_buf += ch
        else:
            if num_buf and unit_buf:
                total += _convert_unit(num_buf, unit_buf)
                matched = True
                num_buf = ""
                unit_buf = ""
            elif num_buf:
                # Bare number — fall through to numeric parse below
                try:
                    return parse_reset_header(float(num_buf))
                except ValueError:
                    return None
    if matched:
        return max(0.0, total)
    return None


def _convert_unit(num: str, unit: str) -> float:
    try:
        value = float(num)
    except ValueError:
        return 0.0
    unit = unit.lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "s":
        return value
    if unit == "m":
        return value * 60.0
    if unit == "h":
        return value * 3600.0
    return 0.0


def extract_retry_after(exc: BaseException) -> float | None:
    """Best-effort extraction of a retry delay from an HTTP exception.

    Inspects common attributes set by ``httpx``, ``openai``, ``anthropic``,
    and similar clients:

    * ``retry_after`` attribute (direct)
    * ``response.headers`` mapping with ``Retry-After`` /
      ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens``
    * ``.headers`` mapping (older clients)
    """
    direct = getattr(exc, "retry_after", None)
    if direct is not None:
        parsed = parse_retry_after(direct)
        if parsed is not None:
            return parsed

    headers = _get_response_headers(exc)
    if not headers:
        return None

    # Standard HTTP
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            parsed = parse_retry_after(headers[key])
            if parsed is not None:
                return parsed

    # OpenAI-style
    candidates: list[float] = []
    for key in (
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
        "X-RateLimit-Reset",
        "X-RateLimit-Reset-Requests",
        "X-RateLimit-Reset-Tokens",
    ):
        if key in headers:
            parsed = parse_reset_header(headers[key])
            if parsed is not None:
                candidates.append(parsed)
    if candidates:
        return max(candidates)
    return None


def _get_response_headers(exc: BaseException) -> Mapping[str, str] | None:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            return headers  # type: ignore[no-any-return]
    headers = getattr(exc, "headers", None)
    if headers is not None:
        return headers  # type: ignore[no-any-return]
    return None


def is_rate_limited(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like an HTTP 429."""
    if isinstance(exc, RateLimitError):
        return True
    # httpx / openai / anthropic style: exc.status_code or exc.response.status_code
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    if status == 429:
        return True
    # Class-name sniffing as a last resort (avoids hard SDK imports).
    name = type(exc).__name__.lower()
    return "ratelimit" in name


def is_server_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a retryable 5xx."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    return isinstance(status, int) and 500 <= status < 600


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for :func:`retry_with_backoff`.

    Attributes:
        max_attempts: Total attempts including the initial try.
        base_delay: First retry delay in seconds.
        max_delay: Upper cap for the computed delay.
        exponential_base: Growth multiplier per attempt.
        jitter: Fractional jitter (0..1) added to each delay.
        retry_on_status: Also retry on these extra status codes (default
            none — ``is_rate_limited`` / ``is_server_error`` already cover
            the standard cases).
    """

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.25
    retry_on_status: tuple[int, ...] = field(default_factory=tuple)


def _backoff_delay(cfg: RetryConfig, attempt: int) -> float:
    """Return the exponential-backoff delay (with jitter) for *attempt*."""
    delay = cfg.base_delay * (cfg.exponential_base ** max(0, attempt))
    delay = min(delay, cfg.max_delay)
    if cfg.jitter > 0:
        # symmetric jitter in [1 - jitter, 1 + jitter]
        lo = max(0.0, 1.0 - cfg.jitter)
        hi = 1.0 + cfg.jitter
        delay *= random.uniform(lo, hi)
    return min(delay, cfg.max_delay)


async def retry_with_backoff(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    config: RetryConfig | None = None,
    provider: str | None = None,
    limiter: ProviderRateLimiter | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Invoke *coro_fn* with 429-aware exponential backoff.

    Args:
        coro_fn: Zero-arg callable producing a *fresh* awaitable per attempt.
        config: Retry configuration; defaults to :class:`RetryConfig`.
        provider: Optional provider name — used for limiter bookkeeping.
        limiter: Optional :class:`ProviderRateLimiter` used to record 429
            hits for observability. Does NOT gate calls; acquire separately.
        sleep: Override for ``asyncio.sleep`` (primarily for tests).

    Returns:
        The result of the first successful attempt.

    Raises:
        The last exception if ``max_attempts`` is exhausted.
    """
    cfg = config or RetryConfig()
    _sleep = sleep or asyncio.sleep

    last_exc: BaseException | None = None
    for attempt in range(cfg.max_attempts):
        try:
            return await coro_fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status is None:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None) if response is not None else None

            rate_limited = is_rate_limited(exc)
            server_error = is_server_error(exc)
            extra_match = isinstance(status, int) and status in cfg.retry_on_status

            if not (rate_limited or server_error or extra_match):
                raise

            if rate_limited and provider and limiter is not None:
                limiter.record_429(provider)

            if attempt >= cfg.max_attempts - 1:
                raise

            hinted = extract_retry_after(exc) if rate_limited else None
            if hinted is not None:
                # Honor server hint, but cap at max_delay and add a small
                # jitter floor so we don't stampede.
                delay = min(hinted, cfg.max_delay)
                if cfg.jitter > 0:
                    delay += random.uniform(0.0, cfg.jitter)
            else:
                delay = _backoff_delay(cfg, attempt)

            logger.info(
                "retry: provider=%s attempt=%d/%d status=%s delay=%.2fs err=%s",
                provider,
                attempt + 1,
                cfg.max_attempts,
                status,
                delay,
                type(exc).__name__,
            )
            await _sleep(delay)

    # Exhausted — mypy-friendly re-raise.
    assert last_exc is not None
    raise last_exc


__all__ = [
    "DEFAULT_FALLBACK",
    "DEFAULT_LIMITS",
    "ProviderRateLimiter",
    "RateLimit",
    "RateLimitError",
    "RetryConfig",
    "TokenBucket",
    "extract_retry_after",
    "get_default_limiter",
    "is_rate_limited",
    "is_server_error",
    "parse_reset_header",
    "parse_retry_after",
    "retry_with_backoff",
    "set_default_limiter",
]
