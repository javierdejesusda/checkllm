"""Tests for checkllm.rate_limit -- per-provider RPM/TPM buckets + 429 retry."""

from __future__ import annotations

import asyncio
import time

import pytest

from checkllm.rate_limit import (
    DEFAULT_LIMITS,
    ProviderRateLimiter,
    RateLimit,
    RateLimitError,
    RetryConfig,
    TokenBucket,
    extract_retry_after,
    is_rate_limited,
    is_server_error,
    parse_reset_header,
    parse_retry_after,
    retry_with_backoff,
)


# ---------------------------------------------------------------------------
# TokenBucket: basic semantics
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            TokenBucket(capacity=0)

    def test_rejects_non_positive_period(self):
        with pytest.raises(ValueError):
            TokenBucket(capacity=10, refill_period=0)

    def test_starts_full(self):
        bucket = TokenBucket(capacity=10, refill_period=60.0)
        assert bucket.available == pytest.approx(10.0)

    def test_try_acquire_success(self):
        bucket = TokenBucket(capacity=10, refill_period=60.0)
        assert bucket.try_acquire(3) is True
        assert bucket.available == pytest.approx(7.0, abs=0.1)

    def test_try_acquire_denied_when_empty(self):
        bucket = TokenBucket(capacity=3, refill_period=60.0)
        assert bucket.try_acquire(3) is True
        assert bucket.try_acquire(1) is False

    async def test_acquire_drains_then_waits_for_refill(self):
        # Capacity 5 refilling every 0.5s => 10 tokens/sec.
        bucket = TokenBucket(capacity=5, refill_period=0.5)
        await bucket.acquire(5)
        assert bucket.available == pytest.approx(0.0, abs=0.2)

        start = time.monotonic()
        await bucket.acquire(2)
        elapsed = time.monotonic() - start
        # Needed ~0.2s to accumulate 2 tokens at 10/s, allow scheduling slack.
        assert 0.15 <= elapsed < 1.0

    async def test_acquire_resumes_after_block(self):
        bucket = TokenBucket(capacity=2, refill_period=0.2)
        await bucket.acquire(2)

        async def take_one() -> float:
            t0 = time.monotonic()
            await bucket.acquire(1)
            return time.monotonic() - t0

        waited = await take_one()
        assert waited >= 0.05

    async def test_acquire_rejects_above_capacity(self):
        bucket = TokenBucket(capacity=5, refill_period=1.0)
        with pytest.raises(ValueError):
            await bucket.acquire(6)

    async def test_refund_caps_at_capacity(self):
        bucket = TokenBucket(capacity=10, refill_period=60.0)
        await bucket.acquire(5)
        bucket.refund(1000)
        assert bucket.available == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# ProviderRateLimiter: registry + dual bucket
# ---------------------------------------------------------------------------


class TestProviderRateLimiter:
    async def test_defaults_cover_known_providers(self):
        limiter = ProviderRateLimiter()
        for name in ("openai", "anthropic", "bedrock", "gemini"):
            state = limiter.get_state(name)
            assert state.limit == DEFAULT_LIMITS[name]

    async def test_unknown_provider_falls_back(self):
        limiter = ProviderRateLimiter()
        state = limiter.get_state("made-up-provider")
        assert state.limit.rpm > 0
        assert state.limit.tpm > 0

    async def test_override_via_constructor(self):
        limiter = ProviderRateLimiter(limits={"openai": RateLimit(rpm=10, tpm=1000)})
        assert limiter.get_state("openai").limit == RateLimit(rpm=10, tpm=1000)

    async def test_configure_replaces(self):
        limiter = ProviderRateLimiter()
        limiter.configure("openai", RateLimit(rpm=7, tpm=770))
        assert limiter.get_state("openai").limit == RateLimit(rpm=7, tpm=770)
        assert limiter.get_state("openai").rpm_bucket.capacity == 7.0

    async def test_acquire_drains_both_buckets(self):
        limiter = ProviderRateLimiter(
            limits={"p": RateLimit(rpm=5, tpm=100)},
        )
        await limiter.acquire("p", est_tokens=40)
        state = limiter.get_state("p")
        assert state.rpm_bucket.available == pytest.approx(4.0, abs=0.1)
        assert state.tpm_bucket.available == pytest.approx(60.0, abs=0.5)

    async def test_acquire_blocks_when_rpm_exhausted(self):
        # 2 rpm over 60s => very slow refill; use short period to test.
        limiter = ProviderRateLimiter()
        limiter.configure("p", RateLimit(rpm=2, tpm=10_000))
        # Shorten refill manually for test speed.
        state = limiter.get_state("p")
        state.rpm_bucket.refill_period = 0.2
        state.rpm_bucket.refill_rate = state.rpm_bucket.capacity / 0.2

        await limiter.acquire("p", est_tokens=1)
        await limiter.acquire("p", est_tokens=1)
        start = time.monotonic()
        await limiter.acquire("p", est_tokens=1)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05

    async def test_release_actual_refunds_overestimate(self):
        limiter = ProviderRateLimiter(
            limits={"p": RateLimit(rpm=100, tpm=1000)},
        )
        await limiter.acquire("p", est_tokens=500)
        state = limiter.get_state("p")
        assert state.tpm_bucket.available == pytest.approx(500.0, abs=1.0)

        # Actual turned out to be 100 — refund 400.
        limiter.release_actual("p", actual_tokens=100)
        assert state.tpm_bucket.available == pytest.approx(900.0, abs=1.0)

    async def test_release_actual_charges_underestimate(self):
        limiter = ProviderRateLimiter(
            limits={"p": RateLimit(rpm=100, tpm=1000)},
        )
        await limiter.acquire("p", est_tokens=100)
        state = limiter.get_state("p")
        before = state.tpm_bucket.available

        limiter.release_actual("p", actual_tokens=300)
        after = state.tpm_bucket.available
        assert after < before

    async def test_large_estimate_is_clamped(self):
        limiter = ProviderRateLimiter(
            limits={"p": RateLimit(rpm=100, tpm=100)},
        )
        # Request more than capacity — should NOT raise; just drain.
        await limiter.acquire("p", est_tokens=1_000_000)
        state = limiter.get_state("p")
        assert state.tpm_bucket.available <= 1.0


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    def test_retry_after_integer_seconds(self):
        assert parse_retry_after("5") == 5.0

    def test_retry_after_float(self):
        assert parse_retry_after("2.5") == 2.5

    def test_retry_after_none(self):
        assert parse_retry_after(None) is None

    def test_retry_after_empty(self):
        assert parse_retry_after("") is None

    def test_retry_after_http_date_future(self):
        import email.utils

        future = time.time() + 10
        date_str = email.utils.formatdate(future, usegmt=True)
        parsed = parse_retry_after(date_str)
        assert parsed is not None
        assert 5 <= parsed <= 15

    def test_retry_after_http_date_past_returns_zero(self):
        import email.utils

        past = time.time() - 100
        date_str = email.utils.formatdate(past, usegmt=True)
        parsed = parse_retry_after(date_str)
        assert parsed == 0.0

    def test_reset_header_seconds(self):
        assert parse_reset_header("3s") == 3.0

    def test_reset_header_ms(self):
        assert parse_reset_header("500ms") == 0.5

    def test_reset_header_compound(self):
        assert parse_reset_header("1m30s") == 90.0

    def test_reset_header_bare_number(self):
        assert parse_reset_header("7") == 7.0

    def test_reset_header_epoch(self):
        future = time.time() + 5
        parsed = parse_reset_header(future)
        assert parsed is not None
        assert 3 <= parsed <= 7


# ---------------------------------------------------------------------------
# Exception inspection
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str]) -> None:
        self.status_code = status
        self.headers = headers


class _FakeHTTPError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.response = _FakeResponse(status, headers or {})
        self.status_code = status


class TestExceptionInspection:
    def test_is_rate_limited_on_429(self):
        assert is_rate_limited(_FakeHTTPError(429))

    def test_is_rate_limited_on_canonical(self):
        assert is_rate_limited(RateLimitError("nope"))

    def test_is_rate_limited_false_on_500(self):
        assert not is_rate_limited(_FakeHTTPError(500))

    def test_is_server_error_on_503(self):
        assert is_server_error(_FakeHTTPError(503))

    def test_is_server_error_false_on_429(self):
        assert not is_server_error(_FakeHTTPError(429))

    def test_extract_retry_after_from_header(self):
        exc = _FakeHTTPError(429, {"Retry-After": "7"})
        assert extract_retry_after(exc) == 7.0

    def test_extract_retry_after_from_ratelimit_reset(self):
        exc = _FakeHTTPError(
            429,
            {"x-ratelimit-reset-requests": "3s", "x-ratelimit-reset-tokens": "5s"},
        )
        assert extract_retry_after(exc) == 5.0

    def test_extract_retry_after_direct_attr(self):
        exc = RateLimitError("nope", retry_after=4.0)
        assert extract_retry_after(exc) == 4.0


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    async def test_success_first_try(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(fn)
        assert result == "ok"
        assert calls == 1

    async def test_retries_on_429(self):
        calls = 0
        sleeps: list[float] = []

        async def fake_sleep(s: float) -> None:
            sleeps.append(s)

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _FakeHTTPError(429, {"Retry-After": "0"})
            return "ok"

        result = await retry_with_backoff(
            fn,
            config=RetryConfig(max_attempts=5, base_delay=0.01, max_delay=0.1, jitter=0.0),
            sleep=fake_sleep,
        )
        assert result == "ok"
        assert calls == 3
        assert len(sleeps) == 2  # 2 retries before success

    async def test_retries_on_5xx(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeHTTPError(502)
            return "ok"

        result = await retry_with_backoff(
            fn,
            config=RetryConfig(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0),
            sleep=lambda _: asyncio.sleep(0),
        )
        assert result == "ok"
        assert calls == 2

    async def test_does_not_retry_on_4xx_non_429(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise _FakeHTTPError(400)

        with pytest.raises(_FakeHTTPError):
            await retry_with_backoff(
                fn,
                config=RetryConfig(max_attempts=5, base_delay=0.0, jitter=0.0),
                sleep=lambda _: asyncio.sleep(0),
            )
        assert calls == 1

    async def test_honors_retry_after_header(self):
        calls = 0
        sleeps: list[float] = []

        async def fake_sleep(s: float) -> None:
            sleeps.append(s)

        async def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeHTTPError(429, {"Retry-After": "3"})
            return "ok"

        await retry_with_backoff(
            fn,
            config=RetryConfig(
                max_attempts=3, base_delay=100.0, max_delay=100.0, jitter=0.0
            ),
            sleep=fake_sleep,
        )
        # Server said 3s; backoff would have been 100s — so we must have used ~3.
        assert len(sleeps) == 1
        assert 2.5 <= sleeps[0] <= 3.5

    async def test_exponential_growth(self):
        calls = 0
        sleeps: list[float] = []

        async def fake_sleep(s: float) -> None:
            sleeps.append(s)

        async def fn():
            nonlocal calls
            calls += 1
            raise _FakeHTTPError(429, {})

        with pytest.raises(_FakeHTTPError):
            await retry_with_backoff(
                fn,
                config=RetryConfig(
                    max_attempts=4,
                    base_delay=0.1,
                    max_delay=100.0,
                    exponential_base=2.0,
                    jitter=0.0,
                ),
                sleep=fake_sleep,
            )

        # Three sleeps before the 4th (final) attempt fails.
        assert len(sleeps) == 3
        # Each sleep should roughly double.
        for i in range(1, len(sleeps)):
            assert sleeps[i] >= sleeps[i - 1] * 1.5

    async def test_exhausts_and_reraises(self):
        async def fn():
            raise _FakeHTTPError(429)

        with pytest.raises(_FakeHTTPError):
            await retry_with_backoff(
                fn,
                config=RetryConfig(max_attempts=2, base_delay=0.0, jitter=0.0),
                sleep=lambda _: asyncio.sleep(0),
            )

    async def test_limiter_records_429_hits(self):
        from checkllm.rate_limit import ProviderRateLimiter

        limiter = ProviderRateLimiter()

        async def fn():
            raise _FakeHTTPError(429)

        with pytest.raises(_FakeHTTPError):
            await retry_with_backoff(
                fn,
                config=RetryConfig(max_attempts=3, base_delay=0.0, jitter=0.0),
                provider="openai",
                limiter=limiter,
                sleep=lambda _: asyncio.sleep(0),
            )
        assert limiter.get_state("openai").rate_429_hits >= 1


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    async def test_submit_judge_retries_on_429(self):
        from checkllm.engines import AsyncEngine
        from checkllm.rate_limit import ProviderRateLimiter

        limiter = ProviderRateLimiter(limits={"openai": RateLimit(rpm=100, tpm=10_000)})
        engine = AsyncEngine(
            max_concurrency=2,
            rate_limiter=limiter,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.0, jitter=0.0),
        )
        calls = 0

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _FakeHTTPError(429, {"Retry-After": "0"})
            return "ok"

        task = await engine.submit_judge("openai", flaky, est_tokens=50)
        result = await task
        assert result == "ok"
        assert calls == 2
        await engine.shutdown()

    async def test_submit_judge_reconciles_actual_tokens(self):
        from checkllm.engines import AsyncEngine
        from checkllm.rate_limit import ProviderRateLimiter

        limiter = ProviderRateLimiter(limits={"openai": RateLimit(rpm=100, tpm=1000)})
        engine = AsyncEngine(
            max_concurrency=1,
            rate_limiter=limiter,
            retry_config=RetryConfig(max_attempts=1, base_delay=0.0, jitter=0.0),
        )

        async def returns_usage():
            return {"tokens": 50}

        task = await engine.submit_judge(
            "openai",
            returns_usage,
            est_tokens=500,
            actual_tokens=lambda r: r["tokens"],
        )
        await task
        state = limiter.get_state("openai")
        # 1000 - 500 (estimate) + (500 - 50) refund = 950
        assert state.tpm_bucket.available == pytest.approx(950.0, abs=1.0)
        await engine.shutdown()

    async def test_config_builds_limiter(self):
        from checkllm.config import CheckllmConfig

        cfg = CheckllmConfig(
            rate_limits={"openai": {"rpm": 7, "tpm": 700}},
        )
        limiter = cfg.build_rate_limiter()
        assert limiter.get_state("openai").limit == RateLimit(rpm=7, tpm=700)

    async def test_config_builds_retry(self):
        from checkllm.config import CheckllmConfig

        cfg = CheckllmConfig(
            retry_max_attempts=7,
            retry_base_delay=0.5,
            retry_max_delay=12.0,
        )
        retry = cfg.build_retry_config()
        assert retry.max_attempts == 7
        assert retry.base_delay == 0.5
        assert retry.max_delay == 12.0
