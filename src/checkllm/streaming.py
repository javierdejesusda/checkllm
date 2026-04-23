"""Streaming evaluation for LLM outputs.

Evaluate LLM outputs as they stream in token by token, running checks at
configurable intervals and supporting early-stop conditions.

The :class:`StreamingEvaluator` natively supports token iterators from any
source, plus convenience adapters for OpenAI (:func:`stream_openai_chat`)
and Anthropic (:func:`stream_anthropic_messages`) async clients.  See
:meth:`StreamingEvaluator.evaluate_provider` for a one-call wrapper that
routes based on the provider type.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

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

    async def evaluate_provider(
        self,
        provider: Any,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        final_checks: bool = True,
        **provider_kwargs: Any,
    ) -> AsyncIterator[StreamingCheckpoint]:
        """Stream from an OpenAI or Anthropic async client and run checks.

        Detects the client type at runtime and routes to the correct
        streaming adapter, yielding checkpoints using the same interval
        behavior as :meth:`evaluate`.

        Args:
            provider: An ``openai.AsyncOpenAI``-compatible client, an
                ``anthropic.AsyncAnthropic``-compatible client, or a
                checkllm judge with a ``stream_evaluate`` method.
            prompt: User message to send.
            system_prompt: Optional system prompt.
            model: Override model name (required for raw SDK clients; judges
                already know their model).
            max_tokens: ``max_tokens`` for the underlying API call (Anthropic
                requires this).
            final_checks: Whether to run a final set of checks when the
                stream ends naturally.
            **provider_kwargs: Extra kwargs forwarded to the SDK streaming
                call (e.g. ``temperature``).

        Yields:
            :class:`StreamingCheckpoint` at each interval and (optionally) at
            the end of the stream.
        """
        token_stream = _provider_token_stream(
            provider,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            **provider_kwargs,
        )
        async for checkpoint in self.evaluate(token_stream, final_checks=final_checks):
            yield checkpoint

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


# ---------------------------------------------------------------------------
# Provider-specific streaming adapters
# ---------------------------------------------------------------------------


def _is_openai_client(provider: Any) -> bool:
    """Return True when ``provider`` looks like an OpenAI async client.

    Identified by the ``chat.completions.create`` attribute path, which is
    unique to OpenAI-shaped clients (the Anthropic SDK has ``messages``, not
    ``chat``).
    """
    cls_name = type(provider).__name__
    # Any real SDK client will be named e.g. ``AsyncOpenAI`` / ``AsyncAzureOpenAI``.
    if cls_name in {"AsyncOpenAI", "AsyncAzureOpenAI"}:
        return True
    chat = getattr(provider, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    return completions is not None and callable(getattr(completions, "create", None))


def _is_anthropic_client(provider: Any) -> bool:
    """Return True when ``provider`` looks like an Anthropic async client.

    Requires both ``messages.stream`` *and* absence of ``chat`` so that
    MagicMock-style duck types aimed at OpenAI don't get mis-detected here.
    """
    cls_name = type(provider).__name__
    if cls_name in {"AsyncAnthropic", "AsyncAnthropicBedrock", "AsyncAnthropicVertex"}:
        return True
    messages = getattr(provider, "messages", None)
    if messages is None:
        return False
    if not callable(getattr(messages, "stream", None)):
        return False
    # Anthropic clients never expose ``chat.completions`` — if this path
    # exists, this is an OpenAI-shaped client with an unrelated ``messages``
    # attribute (common when people pass a ``MagicMock``).
    chat = getattr(provider, "chat", None)
    if chat is not None and getattr(chat, "completions", None) is not None:
        return False
    return True


def _has_stream_evaluate(provider: Any) -> bool:
    """Return True when ``provider`` is a checkllm judge with streaming."""
    return callable(getattr(provider, "stream_evaluate", None))


async def stream_openai_chat(
    client: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Yield content-delta strings from an OpenAI ``chat.completions`` stream.

    Mirrors the interface of :func:`stream_anthropic_messages` so both
    providers can be swapped transparently.

    Args:
        client: An ``openai.AsyncOpenAI``-compatible client.
        prompt: User message.
        system_prompt: Optional system message.
        model: Model name.
        **kwargs: Extra kwargs forwarded to ``chat.completions.create``
            (e.g. ``temperature``, ``max_tokens``).

    Yields:
        String chunks of content as they arrive.
    """
    from checkllm.tracing import propagate_trace_context

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    trace_headers = propagate_trace_context()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_headers=trace_headers or None,
        **kwargs,
    )

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        piece = getattr(delta, "content", None) if delta else None
        if piece:
            yield piece


async def stream_anthropic_messages(
    client: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Yield text-delta strings from an Anthropic ``messages.stream``.

    Mirrors :func:`stream_openai_chat` so both providers share a single
    interface.  Uses the ``content_block_delta`` events emitted by the
    Anthropic SDK.

    Args:
        client: An ``anthropic.AsyncAnthropic``-compatible client.
        prompt: User message.
        system_prompt: Optional system message (passed via ``system=``).
        model: Model name (e.g. ``"claude-sonnet-4-6"``).
        max_tokens: Hard cap on generated tokens.  Required by Anthropic.
        **kwargs: Extra kwargs forwarded to ``messages.stream``
            (e.g. ``temperature``).

    Yields:
        String chunks of text content as they arrive.
    """
    from checkllm.tracing import propagate_trace_context

    trace_headers = propagate_trace_context()
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt or "",
        messages=[{"role": "user", "content": prompt}],
        extra_headers=trace_headers or None,
        **kwargs,
    ) as stream:
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            piece = getattr(delta, "text", None) if delta else None
            if piece:
                yield piece


async def _judge_token_stream(
    judge: Any,
    prompt: str,
    system_prompt: str | None,
) -> AsyncIterator[str]:
    """Yield only the text chunks from a judge's ``stream_evaluate``.

    The judge's terminal ``StreamingJudgeResult`` is discarded so the
    resulting stream matches the raw provider adapters.
    """
    async for item in judge.stream_evaluate(prompt, system_prompt=system_prompt):
        if isinstance(item, str):
            yield item
        # Ignore the terminal StreamingJudgeResult


def _provider_token_stream(
    provider: Any,
    *,
    prompt: str,
    system_prompt: str | None,
    model: str | None,
    max_tokens: int,
    **provider_kwargs: Any,
) -> AsyncIterator[str]:
    """Dispatch to the right streaming adapter for ``provider``.

    Raises:
        TypeError: If ``provider`` is not a supported client / judge type.
        ValueError: If a raw SDK client is passed without a ``model``.
    """
    if _has_stream_evaluate(provider):
        return _judge_token_stream(provider, prompt, system_prompt)

    if _is_anthropic_client(provider):
        if model is None:
            raise ValueError("`model` is required when streaming from a raw Anthropic client")
        return stream_anthropic_messages(
            provider,
            prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            **provider_kwargs,
        )

    if _is_openai_client(provider):
        if model is None:
            raise ValueError("`model` is required when streaming from a raw OpenAI client")
        return stream_openai_chat(
            provider,
            prompt,
            system_prompt=system_prompt,
            model=model,
            **provider_kwargs,
        )

    raise TypeError(
        f"Unsupported provider type {type(provider).__name__!r}: "
        "expected an OpenAI / Anthropic async client or a judge with "
        "`stream_evaluate`."
    )
