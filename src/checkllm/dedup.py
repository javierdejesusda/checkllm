"""In-flight request deduplication for judge calls.

When multiple concurrent ``evaluate()`` calls share the same ``(judge,
prompt, model, temperature)`` key, only one real API call is made.  The
other callers await the same underlying ``asyncio.Future`` and receive the
identical result.

This is different from the judge cache (:mod:`checkllm.cache`):

* The cache serves **finished** requests from a persistent SQLite store.
* The deduplicator coalesces **concurrent** in-flight requests so that a
  burst of parallel calls with the same inputs only pays for one API round
  trip.

The two layers compose naturally: check the cache first (cheap, covers
repeat runs across processes) and then route misses through the
deduplicator so that N workers racing on the same new key do not each
fire the underlying API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger("checkllm.dedup")

T = TypeVar("T")


def make_dedup_key(
    judge: str,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    **extra: Any,
) -> str:
    """Build a stable dedup key from the judge call inputs.

    Args:
        judge: Logical judge identifier (e.g. ``"openai"`` or the class
            name).
        prompt: Full prompt text.
        model: Optional model identifier.
        temperature: Optional sampling temperature.
        **extra: Any other fields that should make the key unique.

    Returns:
        A short deterministic SHA-256 hex digest.
    """
    payload = {
        "judge": judge,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        **extra,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class InFlightDeduplicator:
    """Coalesce concurrent async calls sharing the same key.

    Typical use::

        dedup = InFlightDeduplicator()

        async def run():
            return await dedup.run(key, lambda: judge.evaluate(prompt))

    If N coroutines all call ``run(key, factory)`` before the first one
    completes, only the first call invokes ``factory`` — the rest wait on
    the same shared future.  Exceptions propagate to all waiters.

    Attributes:
        hits: Number of calls that coalesced onto an already-in-flight
            request.
        misses: Number of calls that started a fresh request.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self.hits: int = 0
        self.misses: int = 0

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Run ``factory()`` once per key, sharing the result with all
        concurrent callers.

        Args:
            key: Dedup key.  Use :func:`make_dedup_key` to compute one.
            factory: Zero-arg callable returning an awaitable.  Only the
                first caller for a given ``key`` will invoke it.

        Returns:
            The value resolved by the underlying awaitable.

        Raises:
            Any exception raised by the underlying awaitable propagates
            to every waiter.
        """
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self.hits += 1
                logger.debug("Dedup HIT: %s (hits=%d)", key[:16], self.hits)
                future = existing
                is_owner = False
            else:
                self.misses += 1
                logger.debug("Dedup MISS: %s (misses=%d)", key[:16], self.misses)
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._inflight[key] = future
                is_owner = True

        if is_owner:
            try:
                result = await factory()
            except BaseException as exc:
                # Propagate to waiters and clean up in one atomic step.
                async with self._lock:
                    self._inflight.pop(key, None)
                if not future.done():
                    future.set_exception(exc)
                raise
            else:
                async with self._lock:
                    self._inflight.pop(key, None)
                if not future.done():
                    future.set_result(result)
                return result
        else:
            waited: T = await future
            return waited

    def stats(self) -> dict[str, int]:
        """Return dedup statistics for this instance."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "inflight": len(self._inflight),
        }

    def reset_stats(self) -> None:
        """Zero the hit/miss counters (leaves in-flight futures intact)."""
        self.hits = 0
        self.misses = 0
