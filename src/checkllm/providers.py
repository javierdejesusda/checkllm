"""Multi-provider judge backends for checkllm.

Provides judge implementations for Google Gemini, Azure OpenAI, Ollama,
LiteLLM, and arbitrary HTTP endpoints, plus a ``create_judge`` factory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from checkllm.judge import JudgeBackend, JudgeConfigError, estimate_cost
from checkllm.models import JudgeResponse

# ---------------------------------------------------------------------------
# Token pricing tables (USD per token)
# ---------------------------------------------------------------------------

_GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10 / 1_000_000, 0.40 / 1_000_000),
    "gemini-2.0-pro": (1.25 / 1_000_000, 10.00 / 1_000_000),
    "gemini-1.5-flash": (0.075 / 1_000_000, 0.30 / 1_000_000),
    "gemini-1.5-pro": (1.25 / 1_000_000, 5.00 / 1_000_000),
}

_DEFAULT_PRICE = (5.00 / 1_000_000, 15.00 / 1_000_000)


def _gemini_estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    input_price, output_price = _GEMINI_PRICES.get(model, _DEFAULT_PRICE)
    return prompt_tokens * input_price + completion_tokens * output_price


# ---------------------------------------------------------------------------
# Shared retry decorators
# ---------------------------------------------------------------------------

_transient_retry = retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Shared JSON parsing
# ---------------------------------------------------------------------------

def _parse_judge_json(raw_output: str) -> tuple[float, str]:
    """Parse score and reasoning from a JSON string.

    Returns (score, reasoning).  On failure returns (0.0, <error message>).
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
# GeminiJudge
# ---------------------------------------------------------------------------

class GeminiJudge:
    """Google Gemini-based LLM judge for evaluating outputs."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Google API key not found. Set the GOOGLE_API_KEY environment "
                "variable or pass api_key= to GeminiJudge().\n"
                "  export GOOGLE_API_KEY=AI..."
            )

        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "google-generativeai package not installed. Install it with:\n"
                "  pip install checkllm[gemini]"
            )

        genai.configure(api_key=resolved_key)
        self._genai = genai
        self._gmodel = genai.GenerativeModel(model)

    @_transient_retry
    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        response = await self._gmodel.generate_content_async(
            full_prompt,
            generation_config={"temperature": 0.0},
        )

        raw_output = response.text or ""

        cost = 0.0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            cost = _gemini_estimate_cost(self.model, prompt_tokens, completion_tokens)
        self.last_cost = cost
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    def __repr__(self) -> str:
        return f"GeminiJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# AzureOpenAIJudge
# ---------------------------------------------------------------------------

class AzureOpenAIJudge:
    """Azure-hosted OpenAI judge for evaluating outputs."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-06-01",
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY or "
                "pass api_key= to AzureOpenAIJudge()."
            )

        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            raise JudgeConfigError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT or "
                "pass endpoint= to AzureOpenAIJudge()."
            )

        resolved_deployment = deployment or os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT"
        )
        if not resolved_deployment:
            raise JudgeConfigError(
                "Azure OpenAI deployment not found. Set AZURE_OPENAI_DEPLOYMENT "
                "or pass deployment= to AzureOpenAIJudge()."
            )

        self._deployment = resolved_deployment

        try:
            from openai import AsyncAzureOpenAI  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "openai package not installed. Install it with:\n"
                "  pip install checkllm[openai]"
            )

        self._client = AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version=api_version,
        )

    @_transient_retry
    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._deployment,
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

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    def __repr__(self) -> str:
        return (
            f"AzureOpenAIJudge(model={self.model!r}, "
            f"deployment={self._deployment!r}, "
            f"total_cost=${self.total_cost:.4f})"
        )


# ---------------------------------------------------------------------------
# OllamaJudge
# ---------------------------------------------------------------------------

class OllamaJudge:
    """Judge backed by a local Ollama instance (pure HTTP, no extra packages)."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self._timeout = timeout
        self._host = (
            host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        )

    @_transient_retry
    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            try:
                import aiohttp  # type: ignore[import-untyped]
            except ImportError:
                raise JudgeConfigError(
                    "Neither httpx nor aiohttp installed. Install one with:\n"
                    "  pip install httpx"
                )
            return await self._evaluate_aiohttp(prompt, system_prompt)

        return await self._evaluate_httpx(prompt, system_prompt)

    async def _evaluate_httpx(
        self, prompt: str, system_prompt: str | None
    ) -> JudgeResponse:
        import httpx

        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        url = f"{self._host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_output = data.get("response", "")
        cost = 0.0
        self.last_cost = cost
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    async def _evaluate_aiohttp(
        self, prompt: str, system_prompt: str | None
    ) -> JudgeResponse:
        import aiohttp

        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        url = f"{self._host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw_output = data.get("response", "")
        cost = 0.0
        self.last_cost = cost
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    def __repr__(self) -> str:
        return (
            f"OllamaJudge(model={self.model!r}, host={self._host!r})"
        )


# ---------------------------------------------------------------------------
# LiteLLMJudge
# ---------------------------------------------------------------------------

class LiteLLMJudge:
    """Judge that delegates to LiteLLM for 100+ model providers."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self._api_key = api_key
        self._api_base = api_base
        self._extra_kwargs = kwargs

        try:
            import litellm  # type: ignore[import-untyped] # noqa: F401
        except ImportError:
            raise JudgeConfigError(
                "litellm package not installed. Install it with:\n"
                "  pip install checkllm[litellm]"
            )

    @_transient_retry
    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        import litellm  # type: ignore[import-untyped]

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self._api_key:
            call_kwargs["api_key"] = self._api_key
        if self._api_base:
            call_kwargs["api_base"] = self._api_base
        call_kwargs.update(self._extra_kwargs)

        response = await litellm.acompletion(**call_kwargs)

        raw_output = response.choices[0].message.content or ""

        # Use litellm's built-in cost tracking
        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # Fall back to our own estimate if litellm cost tracking fails
            if response.usage:
                cost = estimate_cost(
                    self.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
        self.last_cost = cost
        self.total_cost += cost

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    def __repr__(self) -> str:
        return f"LiteLLMJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# CustomHTTPJudge
# ---------------------------------------------------------------------------

# Default response parser: expects {"score": float, "reasoning": str}
ResponseParser = Callable[[dict[str, Any]], tuple[float, str]]


def _default_response_parser(data: dict[str, Any]) -> tuple[float, str]:
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    reasoning = str(data.get("reasoning", ""))
    return score, reasoning


class CustomHTTPJudge:
    """Judge that calls an arbitrary REST endpoint."""

    def __init__(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        response_parser: ResponseParser | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = f"custom:{url}"
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self._url = url
        self._method = method.upper()
        self._headers = headers or {}
        self._response_parser = response_parser or _default_response_parser
        self._timeout = timeout
        self._max_retries = max_retries

    async def evaluate(
        self, prompt: str, system_prompt: str | None = None
    ) -> JudgeResponse:
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "httpx package not installed. Install it with:\n"
                "  pip install httpx"
            )

        payload = {"prompt": prompt, "system_prompt": system_prompt}

        last_exc: Exception | None = None
        for _attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.request(
                        self._method,
                        self._url,
                        json=payload,
                        headers=self._headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except (httpx.TimeoutException, httpx.ConnectError, OSError) as exc:
                last_exc = exc
                continue
        else:
            raise ConnectionError(
                f"Failed to reach {self._url} after {self._max_retries} attempts"
            ) from last_exc

        raw_output = json.dumps(data)
        cost = 0.0
        self.last_cost = cost
        self.total_cost += cost

        try:
            score, reasoning = self._response_parser(data)
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.0
            reasoning = f"Failed to parse response: {raw_output[:200]}"

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
        )

    def __repr__(self) -> str:
        return f"CustomHTTPJudge(url={self._url!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_judge(backend: str, **kwargs: Any) -> JudgeBackend:
    """Create a judge backend by name.

    Parameters
    ----------
    backend:
        One of "openai", "anthropic", "gemini", "azure", "ollama",
        "litellm", or "custom".
    **kwargs:
        Forwarded to the backend constructor.

    Returns
    -------
    JudgeBackend
    """
    from checkllm.judge import AnthropicJudge, OpenAIJudge

    _backends: dict[str, type] = {
        "openai": OpenAIJudge,
        "anthropic": AnthropicJudge,
        "gemini": GeminiJudge,
        "azure": AzureOpenAIJudge,
        "ollama": OllamaJudge,
        "litellm": LiteLLMJudge,
        "custom": CustomHTTPJudge,
    }

    cls = _backends.get(backend)
    if cls is None:
        supported = ", ".join(sorted(_backends))
        raise ValueError(
            f"Unknown backend {backend!r}. Supported: {supported}"
        )

    return cls(**kwargs)  # type: ignore[return-value]
