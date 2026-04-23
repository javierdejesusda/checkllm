from __future__ import annotations

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from checkllm.models import JudgeResponse

if TYPE_CHECKING:
    from checkllm.multimodal import ImagePayload

# ---------------------------------------------------------------------------
# Token pricing tables (USD per token)
# ---------------------------------------------------------------------------

_OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4-turbo": (10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-4": (30.00 / 1_000_000, 60.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}

_ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-6": (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.00 / 1_000_000),
    "claude-3-5-sonnet-20241022": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-3-haiku-20240307": (0.25 / 1_000_000, 1.25 / 1_000_000),
}

# DeepSeek published pricing (USD per token) as of early 2026.
# TODO: review periodically — see https://api-docs.deepseek.com/quick_start/pricing
_DEEPSEEK_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27 / 1_000_000, 1.10 / 1_000_000),
    "deepseek-reasoner": (0.55 / 1_000_000, 2.19 / 1_000_000),
}

_DEFAULT_PRICE = (5.00 / 1_000_000, 15.00 / 1_000_000)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost for an API call based on token usage."""
    prices = {**_OPENAI_PRICES, **_ANTHROPIC_PRICES, **_DEEPSEEK_PRICES}
    input_price, output_price = prices.get(model, _DEFAULT_PRICE)
    return prompt_tokens * input_price + completion_tokens * output_price


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class JudgeConfigError(Exception):
    """Raised when a judge backend is misconfigured (e.g., missing API key)."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeBackend(Protocol):
    """Protocol for LLM judge backends."""

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse: ...


EarlyStopFn = Callable[[str], bool]


class StreamingJudgeResult:
    """Result wrapper emitted as the last chunk of a streaming judge call.

    Streaming judges yield plain string chunks until generation completes,
    then yield a single instance of this class containing the parsed
    ``JudgeResponse`` and the aggregated raw text.
    """

    def __init__(
        self,
        response: JudgeResponse,
        aggregated_text: str,
        stopped_early: bool = False,
    ) -> None:
        self.response = response
        self.aggregated_text = aggregated_text
        self.stopped_early = stopped_early

    def __repr__(self) -> str:
        return (
            f"StreamingJudgeResult(score={self.response.score:.2f}, "
            f"stopped_early={self.stopped_early})"
        )


def _parse_judge_json(raw_output: str) -> tuple[float, str]:
    """Parse score and reasoning from a JSON string.

    Args:
        raw_output: Raw text returned by the judge.

    Returns:
        A tuple ``(score, reasoning)``.  On failure returns
        ``(0.0, "<error message>")``.
    """
    try:
        parsed = json.loads(raw_output)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = str(parsed.get("reasoning", ""))
        return score, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0, f"Failed to parse judge response: {raw_output[:200]}"


# ---------------------------------------------------------------------------
# Retry decorator for transient API errors
# ---------------------------------------------------------------------------


def _make_retry() -> Any:
    """Create a retry decorator for transient API failures."""
    try:
        from openai import APITimeoutError, RateLimitError, APIConnectionError

        transient: tuple[type[BaseException], ...] = (
            APITimeoutError,
            RateLimitError,
            APIConnectionError,
        )
    except ImportError:
        transient = (TimeoutError, ConnectionError)

    return retry(
        retry=retry_if_exception_type(transient),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )


_api_retry: Any = _make_retry()


# ---------------------------------------------------------------------------
# OpenAI judge
# ---------------------------------------------------------------------------


class OpenAIJudge:
    """OpenAI-based LLM judge for evaluating outputs."""

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise JudgeConfigError(
                "openai package not installed — pip install checkllm[openai]"
                " (or pip install checkllm[all] for everything)"
            )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to OpenAIJudge().\n"
                "  export OPENAI_API_KEY=sk-..."
            )

        self._client = AsyncOpenAI(api_key=resolved_key)

    @_api_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content or ""

        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = int(response.usage.prompt_tokens or 0)
            output_tokens = int(response.usage.completion_tokens or 0)
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        try:
            parsed = json.loads(raw_output)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.0
            reasoning = f"Failed to parse judge response: {raw_output[:200]}"

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="openai",
        )

    @_api_retry
    async def evaluate_with_images(
        self,
        prompt: str,
        images: Sequence["ImagePayload"],
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        """Call the vision model with one or more images alongside ``prompt``.

        Args:
            prompt: The text portion of the user message.
            images: Normalized image payloads to include before the text.
            system_prompt: Optional system message.

        Returns:
            A ``JudgeResponse`` parsed from the model's JSON output.
        """
        from checkllm.multimodal import to_openai_content

        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict[str, object]] = [to_openai_content(img) for img in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content or ""

        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = int(response.usage.prompt_tokens or 0)
            output_tokens = int(response.usage.completion_tokens or 0)
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        try:
            parsed = json.loads(raw_output)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.0
            reasoning = f"Failed to parse judge response: {raw_output[:200]}"

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="openai",
        )

    async def stream_evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        on_token: EarlyStopFn | None = None,
    ) -> AsyncIterator[str | StreamingJudgeResult]:
        """Stream partial text chunks from the model.

        Yields each content delta as a string, then a final
        :class:`StreamingJudgeResult` carrying the parsed
        :class:`JudgeResponse`.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system-role prompt.
            on_token: Optional callback invoked with the accumulated text
                after every chunk.  Returning ``True`` aborts streaming and
                the generator yields a final result immediately.

        Yields:
            ``str`` chunks, followed by a single ``StreamingJudgeResult``.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        aggregated: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        stopped_early = False

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if not piece:
                continue

            aggregated.append(piece)
            yield piece

            if on_token is not None:
                try:
                    if on_token("".join(aggregated)):
                        stopped_early = True
                        break
                except Exception:
                    # A callback failure should never crash the stream.
                    pass

        raw_output = "".join(aggregated)
        cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(prompt_tokens)
        self.last_output_tokens = int(completion_tokens)
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)
        response = JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens),
            model=self.model,
            provider="openai",
        )
        yield StreamingJudgeResult(
            response=response,
            aggregated_text=raw_output,
            stopped_early=stopped_early,
        )

    def __repr__(self) -> str:
        return f"OpenAIJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# Anthropic judge
# ---------------------------------------------------------------------------


class AnthropicJudge:
    """Anthropic Claude-based LLM judge for evaluating outputs."""

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key= to AnthropicJudge().\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise JudgeConfigError(
                "anthropic package not installed. Install it with:\n"
                "  pip install checkllm[anthropic]"
            )

        self._client = AsyncAnthropic(api_key=resolved_key)

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        system = system_prompt or ""
        full_prompt = (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        )

        raw_output = response.content[0].text if response.content else ""

        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = int(response.usage.input_tokens or 0)
            output_tokens = int(response.usage.output_tokens or 0)
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        try:
            parsed = json.loads(raw_output)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.0
            reasoning = f"Failed to parse judge response: {raw_output[:200]}"

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="anthropic",
        )

    async def evaluate_with_images(
        self,
        prompt: str,
        images: Sequence["ImagePayload"],
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        """Call the Claude vision model with images alongside ``prompt``.

        Args:
            prompt: The text portion of the user message.
            images: Normalized image payloads to include before the text.
            system_prompt: Optional system message.

        Returns:
            A ``JudgeResponse`` parsed from the model's JSON output.
        """
        from checkllm.multimodal import to_anthropic_content

        system = system_prompt or ""
        full_prompt = (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )
        content: list[dict[str, object]] = [to_anthropic_content(img) for img in images]
        content.append({"type": "text", "text": full_prompt})

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

        raw_output = response.content[0].text if response.content else ""

        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = int(response.usage.input_tokens or 0)
            output_tokens = int(response.usage.output_tokens or 0)
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        try:
            parsed = json.loads(raw_output)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.0
            reasoning = f"Failed to parse judge response: {raw_output[:200]}"

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="anthropic",
        )

    async def stream_evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        on_token: EarlyStopFn | None = None,
    ) -> AsyncIterator[str | StreamingJudgeResult]:
        """Stream partial text chunks from the Anthropic messages API.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system-role prompt.
            on_token: Optional callback invoked with the accumulated text
                after every chunk.  Returning ``True`` aborts streaming and
                the generator yields a final result immediately.

        Yields:
            ``str`` chunks, followed by a single ``StreamingJudgeResult``.
        """
        system = system_prompt or ""
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with JSON only: "
            '{"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        aggregated: list[str] = []
        stopped_early = False
        input_tokens = 0
        output_tokens = 0

        async with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    piece = getattr(delta, "text", None) if delta else None
                    if not piece:
                        continue
                    aggregated.append(piece)
                    yield piece

                    if on_token is not None:
                        try:
                            if on_token("".join(aggregated)):
                                stopped_early = True
                                break
                        except Exception:
                            pass
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0)
                elif event_type == "message_start":
                    msg = getattr(event, "message", None)
                    usage = getattr(msg, "usage", None) if msg else None
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0)

        raw_output = "".join(aggregated)
        cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)
        response = JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            model=self.model,
            provider="anthropic",
        )
        yield StreamingJudgeResult(
            response=response,
            aggregated_text=raw_output,
            stopped_early=stopped_early,
        )

    def __repr__(self) -> str:
        return f"AnthropicJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# DeepSeek judge
# ---------------------------------------------------------------------------

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekJudge:
    """Native DeepSeek judge using the OpenAI-compatible SDK.

    DeepSeek exposes an OpenAI-compatible chat-completions endpoint but adds
    its own ``reasoning_content`` field for the ``deepseek-reasoner`` model
    (DeepSeek-R1 family).  This judge surfaces both the standard ``content``
    and ``reasoning_content`` fields and applies DeepSeek-specific pricing.

    Supported models:
        * ``deepseek-chat`` — general-purpose chat model.
        * ``deepseek-reasoner`` — DeepSeek-R1 reasoning model.  Its extended
          chain-of-thought is exposed via
          :attr:`JudgeResponse.raw_output` (appended as a fenced block) and
          via the judge's :attr:`last_reasoning_content` attribute.
    """

    provider: str = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_reasoning_content: str | None = None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise JudgeConfigError(
                "openai package not installed — pip install checkllm[openai]"
                " (the DeepSeek judge uses the OpenAI SDK pointed at the"
                " DeepSeek endpoint)."
            )

        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "DeepSeek API key not found. Set the DEEPSEEK_API_KEY "
                "environment variable or pass api_key= to DeepSeekJudge().\n"
                "  export DEEPSEEK_API_KEY=sk-..."
            )

        self._api_key = resolved_key
        self._base_url = (base_url or _DEEPSEEK_BASE_URL).rstrip("/")
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=self._base_url)

    @_api_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Call the DeepSeek chat-completions endpoint.

        Args:
            prompt: User message.
            system_prompt: Optional system message.

        Returns:
            Parsed :class:`JudgeResponse`.  For reasoning models the
            extracted chain-of-thought is stored on
            :attr:`last_reasoning_content` and appended to
            :attr:`JudgeResponse.raw_output`.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Respond with JSON only: "
                    '{"score": <float 0-1>, "reasoning": "<explanation>"}'
                ),
            }
        )

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        self.last_reasoning_content = reasoning_content

        raw_output = content
        if reasoning_content:
            raw_output = f"{content}\n\n<reasoning>\n{reasoning_content}\n</reasoning>"

        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = int(response.usage.prompt_tokens or 0)
            output_tokens = int(response.usage.completion_tokens or 0)
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(input_tokens)
        self.last_output_tokens = int(output_tokens)
        self.total_cost += cost

        score, reasoning = _parse_judge_json(content)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="deepseek",
        )

    async def stream_evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        on_token: EarlyStopFn | None = None,
    ) -> AsyncIterator[str | StreamingJudgeResult]:
        """Stream partial text chunks from DeepSeek.

        Yields content deltas as strings.  For ``deepseek-reasoner`` the
        reasoning-content deltas are emitted too (prefixed with a zero-width
        marker internally and tracked on :attr:`last_reasoning_content`).

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            on_token: Optional early-stop callback.  If it returns ``True``
                on any accumulated visible content, streaming halts.

        Yields:
            ``str`` chunks, followed by a final ``StreamingJudgeResult``.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Respond with JSON only: "
                    '{"score": <float 0-1>, "reasoning": "<explanation>"}'
                ),
            }
        )

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        visible: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        stopped_early = False

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)

            piece = getattr(delta, "content", None)
            if piece:
                visible.append(piece)
                yield piece

                if on_token is not None:
                    try:
                        if on_token("".join(visible)):
                            stopped_early = True
                            break
                    except Exception:
                        pass

        raw_content = "".join(visible)
        reasoning_content = "".join(reasoning_parts) or None
        self.last_reasoning_content = reasoning_content

        raw_output = raw_content
        if reasoning_content:
            raw_output = f"{raw_content}\n\n<reasoning>\n{reasoning_content}\n</reasoning>"

        cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
        self.last_cost = cost
        self.last_input_tokens = int(prompt_tokens)
        self.last_output_tokens = int(completion_tokens)
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_content)
        response = JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens),
            model=self.model,
            provider="deepseek",
        )
        yield StreamingJudgeResult(
            response=response,
            aggregated_text=raw_output,
            stopped_early=stopped_early,
        )

    def __repr__(self) -> str:
        return f"DeepSeekJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"
