"""Streaming evaluation for LLM outputs.

Evaluate LLM outputs as they stream in token by token, running checks at
configurable intervals and supporting early-stop conditions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

from pydantic import BaseModel, Field

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.streaming")


class StreamingCheckpoint(BaseModel):
    """A checkpoint during streaming evaluation.

    Emitted at regular intervals as tokens arrive, containing the current
    state of all checks run against the partial output so far.
    """

    tokens_received: int
    partial_output: str
    checks_run: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    elapsed_ms: int = 0
    results: list[CheckResult] = Field(default_factory=list)


class StreamingEvaluator:
    """Evaluate LLM outputs as they stream in, token by token.

    Accumulates tokens and runs registered checks at configurable intervals.
    Supports both synchronous and asynchronous check functions, as well as
    early-stop conditions that can halt streaming.

    Usage::

        evaluator = StreamingEvaluator()
        evaluator.add_check("length", lambda text: check.max_tokens(text, 500))
        evaluator.add_check("no_pii", lambda text: check.no_pii(text))
        evaluator.add_early_stop(lambda text: "STOP" in text)

        async for checkpoint in evaluator.evaluate(token_stream):
            print(f"Tokens: {checkpoint.tokens_received}, Passed: {checkpoint.checks_passed}")

    Parameters
    ----------
    check_interval:
        Run checks every N tokens received. Lower values give more frequent
        feedback but increase overhead.
    """

    def __init__(self, check_interval: int = 50) -> None:
        if check_interval < 1:
            raise ValueError("check_interval must be >= 1")

        self._check_interval = check_interval
        self._sync_checks: list[tuple[str, Callable[[str], CheckResult]]] = []
        self._async_checks: list[tuple[str, Callable[[str], Awaitable[CheckResult]]]] = []
        self._early_stops: list[Callable[[str], bool]] = []

    def add_check(self, name: str, check_fn: Callable[[str], CheckResult]) -> None:
        """Register a synchronous check function.

        Parameters
        ----------
        name:
            Human-readable name for logging/identification.
        check_fn:
            A callable that takes the accumulated text and returns a
            ``CheckResult``.
        """
        self._sync_checks.append((name, check_fn))

    def add_async_check(self, name: str, check_fn: Callable[[str], Awaitable[CheckResult]]) -> None:
        """Register an asynchronous check function.

        Parameters
        ----------
        name:
            Human-readable name for logging/identification.
        check_fn:
            An async callable that takes the accumulated text and returns a
            ``CheckResult``.
        """
        self._async_checks.append((name, check_fn))

    def add_early_stop(self, condition: Callable[[str], bool]) -> None:
        """Register an early-stop condition.

        If any registered condition returns ``True``, streaming evaluation
        will stop immediately and yield a final checkpoint.

        Parameters
        ----------
        condition:
            A callable that takes the accumulated text and returns ``True``
            if streaming should stop.
        """
        self._early_stops.append(condition)

    async def _run_checks(self, text: str) -> list[CheckResult]:
        """Run all registered checks (sync and async) against the text."""
        results: list[CheckResult] = []

        # Run sync checks
        for name, check_fn in self._sync_checks:
            try:
                result = check_fn(text)
                results.append(result)
            except Exception as exc:
                logger.warning("Sync check '%s' raised an exception: %s", name, exc)
                results.append(
                    CheckResult(
                        passed=False,
                        score=0.0,
                        reasoning=f"Check '{name}' failed with error: {exc}",
                        cost=0.0,
                        latency_ms=0,
                        metric_name=name,
                    )
                )

        # Run async checks concurrently
        if self._async_checks:
            async_tasks: list[tuple[str, asyncio.Task[CheckResult]]] = []
            for name, check_fn in self._async_checks:
                task = asyncio.create_task(check_fn(text))
                async_tasks.append((name, task))

            for name, task in async_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as exc:
                    logger.warning("Async check '%s' raised an exception: %s", name, exc)
                    results.append(
                        CheckResult(
                            passed=False,
                            score=0.0,
                            reasoning=f"Check '{name}' failed with error: {exc}",
                            cost=0.0,
                            latency_ms=0,
                            metric_name=name,
                        )
                    )

        return results

    def _check_early_stop(self, text: str) -> bool:
        """Return True if any early-stop condition is met."""
        for condition in self._early_stops:
            try:
                if condition(text):
                    return True
            except Exception as exc:
                logger.warning("Early stop condition raised an exception: %s", exc)
        return False

    def _build_checkpoint(
        self,
        tokens_received: int,
        partial_output: str,
        results: list[CheckResult],
        elapsed_ms: int,
    ) -> StreamingCheckpoint:
        """Build a checkpoint from current state."""
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        return StreamingCheckpoint(
            tokens_received=tokens_received,
            partial_output=partial_output,
            checks_run=len(results),
            checks_passed=passed,
            checks_failed=failed,
            elapsed_ms=elapsed_ms,
            results=list(results),
        )

    async def evaluate(
        self,
        token_stream: AsyncIterator[str],
        final_checks: bool = True,
    ) -> AsyncIterator[StreamingCheckpoint]:
        """Evaluate a streaming token source.

        Accumulates tokens from the async iterator, running checks at each
        interval and yielding checkpoints.

        Parameters
        ----------
        token_stream:
            An async iterator that yields string tokens.
        final_checks:
            If ``True``, run all checks one last time after the stream ends
            and yield a final checkpoint.

        Yields
        ------
        StreamingCheckpoint
            A checkpoint at each check interval and (optionally) at the end.
        """
        buffer: list[str] = []
        token_count = 0
        start_time = time.monotonic()
        last_results: list[CheckResult] = []
        stopped_early = False

        async for token in token_stream:
            buffer.append(token)
            token_count += 1

            accumulated = "".join(buffer)

            # Check early-stop conditions
            if self._check_early_stop(accumulated):
                logger.info("Early stop triggered at %d tokens", token_count)
                stopped_early = True
                # Run checks one last time before stopping
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                last_results = await self._run_checks(accumulated)
                yield self._build_checkpoint(token_count, accumulated, last_results, elapsed_ms)
                return

            # Run checks at interval
            if token_count % self._check_interval == 0:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                last_results = await self._run_checks(accumulated)
                checkpoint = self._build_checkpoint(
                    token_count, accumulated, last_results, elapsed_ms
                )
                logger.debug(
                    "Checkpoint at %d tokens: %d passed, %d failed",
                    token_count,
                    checkpoint.checks_passed,
                    checkpoint.checks_failed,
                )
                yield checkpoint

        # Stream ended naturally — run final checks if requested
        if final_checks and not stopped_early:
            accumulated = "".join(buffer)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            # Only run final checks if we haven't just run them
            if token_count % self._check_interval != 0 or token_count == 0:
                last_results = await self._run_checks(accumulated)

            yield self._build_checkpoint(token_count, accumulated, last_results, elapsed_ms)

    async def evaluate_string_chunks(self, chunks: list[str]) -> StreamingCheckpoint:
        """Convenience method for testing with pre-split chunks.

        Feeds each chunk as a token through the evaluator and returns the
        final checkpoint.

        Parameters
        ----------
        chunks:
            List of string chunks to evaluate.

        Returns
        -------
        StreamingCheckpoint
            The final checkpoint after all chunks have been processed.
        """

        async def _chunk_iter() -> AsyncIterator[str]:
            for chunk in chunks:
                yield chunk

        final_checkpoint: StreamingCheckpoint | None = None
        async for checkpoint in self.evaluate(_chunk_iter(), final_checks=True):
            final_checkpoint = checkpoint

        if final_checkpoint is None:
            # No tokens were provided
            return StreamingCheckpoint(
                tokens_received=0,
                partial_output="",
                checks_run=0,
                checks_passed=0,
                checks_failed=0,
                elapsed_ms=0,
                results=[],
            )

        return final_checkpoint
