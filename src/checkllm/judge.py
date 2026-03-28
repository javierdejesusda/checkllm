from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from checkllm.models import JudgeResponse


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
        )
