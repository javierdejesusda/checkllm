from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from checkllm.models import JudgeResponse

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

_DEFAULT_PRICE = (5.00 / 1_000_000, 15.00 / 1_000_000)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost for an API call based on token usage."""
    prices = {**_OPENAI_PRICES, **_ANTHROPIC_PRICES}
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

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse: ...


# ---------------------------------------------------------------------------
# Retry decorator for transient API errors
# ---------------------------------------------------------------------------

def _make_retry():
    """Create a retry decorator for transient API failures."""
    try:
        from openai import APITimeoutError, RateLimitError, APIConnectionError
        transient = (APITimeoutError, RateLimitError, APIConnectionError)
    except ImportError:
        transient = (TimeoutError, ConnectionError)

    return retry(
        retry=retry_if_exception_type(transient),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )

_api_retry = _make_retry()


# ---------------------------------------------------------------------------
# OpenAI judge
# ---------------------------------------------------------------------------

class OpenAIJudge:
    """OpenAI-based LLM judge for evaluating outputs."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to OpenAIJudge().\n"
                "  export OPENAI_API_KEY=sk-..."
            )

        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=resolved_key)

    @_api_retry
    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
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
        if response.usage:
            cost = estimate_cost(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        self.last_cost = cost
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
        )

    def __repr__(self) -> str:
        return f"OpenAIJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# Anthropic judge
# ---------------------------------------------------------------------------

class AnthropicJudge:
    """Anthropic Claude-based LLM judge for evaluating outputs."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

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

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        system = system_prompt or ""
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with JSON only: {\"score\": <float 0-1>, \"reasoning\": \"<explanation>\"}"
        )

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        )

        raw_output = response.content[0].text if response.content else ""

        cost = 0.0
        if response.usage:
            cost = estimate_cost(
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        self.last_cost = cost
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
        )

    def __repr__(self) -> str:
        return f"AnthropicJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"
