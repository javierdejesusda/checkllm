from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from checkllm.models import JudgeResponse

# Approximate per-token pricing (USD) for common models.
# Users can override by subclassing or setting custom pricing.
_TOKEN_PRICES: dict[str, tuple[float, float]] = {
    # (input_price_per_token, output_price_per_token)
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4-turbo": (10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-4": (30.00 / 1_000_000, 60.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}

# Default fallback pricing if model not in the table
_DEFAULT_PRICE = (5.00 / 1_000_000, 15.00 / 1_000_000)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost for an API call based on token usage."""
    input_price, output_price = _TOKEN_PRICES.get(model, _DEFAULT_PRICE)
    return prompt_tokens * input_price + completion_tokens * output_price


@runtime_checkable
class JudgeBackend(Protocol):
    """Protocol for LLM judge backends."""

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse: ...


class OpenAIJudge:
    """OpenAI-based LLM judge for evaluating outputs."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

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

        # Track cost from usage
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
