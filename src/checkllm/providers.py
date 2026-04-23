"""Multi-provider judge backends for checkllm.

Provides judge implementations for Google Gemini, Google Vertex AI,
Azure OpenAI, Ollama, LiteLLM, Cohere, Mistral, AWS Bedrock, and
OpenAI-compatible endpoints (DeepSeek, Groq, Together, Fireworks,
Perplexity, vLLM, OpenRouter, X.AI), plus a ``create_judge`` factory.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from checkllm.multimodal import ImagePayload

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from checkllm.judge import JudgeBackend, JudgeConfigError, estimate_cost
from checkllm.models import JudgeResponse
from checkllm.tracing import propagate_trace_context

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


def _gemini_estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
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
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
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

    async def evaluate_with_images(
        self,
        prompt: str,
        images: Sequence["ImagePayload"],
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        """Call the Gemini vision model with images alongside ``prompt``.

        Args:
            prompt: The text portion of the user turn.
            images: Normalized image payloads to send as inline parts.
            system_prompt: Optional preamble prepended to ``prompt``.

        Returns:
            A ``JudgeResponse`` parsed from the model's JSON output.
        """
        from checkllm.multimodal import to_gemini_part

        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += (
            f"{prompt}\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        )

        parts: list[Any] = [to_gemini_part(img) for img in images]
        parts.append({"text": full_prompt})

        response = await self._gmodel.generate_content_async(
            parts,
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

        resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
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
                "openai package not installed. Install it with:\n  pip install checkllm[openai]"
            )

        self._client = AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version=api_version,
        )

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        trace_headers = propagate_trace_context()
        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers=trace_headers or None,
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
        self._host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        try:
            import httpx  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            try:
                import aiohttp  # type: ignore[import-untyped]  # noqa: F401
            except ImportError:
                raise JudgeConfigError(
                    "Neither httpx nor aiohttp installed. Install one with:\n  pip install httpx"
                )
            return await self._evaluate_aiohttp(prompt, system_prompt)

        return await self._evaluate_httpx(prompt, system_prompt)

    async def _evaluate_httpx(self, prompt: str, system_prompt: str | None) -> JudgeResponse:
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

        trace_headers = propagate_trace_context()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=trace_headers or None)
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

    async def _evaluate_aiohttp(self, prompt: str, system_prompt: str | None) -> JudgeResponse:
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

        trace_headers = propagate_trace_context()
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=trace_headers or None) as resp:
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
        return f"OllamaJudge(model={self.model!r}, host={self._host!r})"


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
                "litellm package not installed. Install it with:\n  pip install checkllm[litellm]"
            )

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
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

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "httpx package not installed. Install it with:\n  pip install httpx"
            )

        payload = {"prompt": prompt, "system_prompt": system_prompt}

        merged_headers = propagate_trace_context(self._headers)

        last_exc: Exception | None = None
        for _attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.request(
                        self._method,
                        self._url,
                        json=payload,
                        headers=merged_headers,
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


class OpenAICompatibleJudge:
    """Judge backend for any OpenAI-compatible API endpoint.

    Uses httpx to call the ``/chat/completions`` endpoint directly, avoiding
    the need for provider-specific SDKs.  Subclasses only need to set
    defaults for *model*, *api_key*, and *base_url*.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._extra_headers = extra_headers or {}

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Send a chat-completion request and parse the judge response."""
        import httpx

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

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers = propagate_trace_context(headers)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        url = f"{self._base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw_output = data["choices"][0]["message"]["content"] or ""

        cost = 0.0
        usage = data.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
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
            f"{type(self).__name__}(model={self.model!r}, "
            f"base_url={self._base_url!r}, "
            f"total_cost=${self.total_cost:.4f})"
        )


# DeepSeekJudge has moved to :mod:`checkllm.judge`.  It now uses the native
# OpenAI SDK (pointed at DeepSeek's endpoint) with DeepSeek-specific pricing
# and ``reasoning_content`` support for ``deepseek-reasoner``.  It is
# re-exported here for backward compatibility so ``from checkllm.providers
# import DeepSeekJudge`` keeps working.
from checkllm.judge import DeepSeekJudge  # noqa: E402,F401


class GroqJudge(OpenAICompatibleJudge):
    """Groq judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Groq API key not found. Set GROQ_API_KEY or pass api_key= to GroqJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )


class TogetherJudge(OpenAICompatibleJudge):
    """Together AI judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Together API key not found. Set TOGETHER_API_KEY or "
                "pass api_key= to TogetherJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://api.together.xyz/v1",
            **kwargs,
        )


class FireworksJudge(OpenAICompatibleJudge):
    """Fireworks AI judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Fireworks API key not found. Set FIREWORKS_API_KEY or "
                "pass api_key= to FireworksJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://api.fireworks.ai/inference/v1",
            **kwargs,
        )


class PerplexityJudge(OpenAICompatibleJudge):
    """Perplexity AI judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama-3.1-sonar-large-128k-online",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Perplexity API key not found. Set PERPLEXITY_API_KEY or "
                "pass api_key= to PerplexityJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://api.perplexity.ai",
            **kwargs,
        )


class VLLMJudge(OpenAICompatibleJudge):
    """Judge backed by a local vLLM server (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str = "default",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_url = base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1"
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=resolved_url,
            **kwargs,
        )


class OpenRouterJudge(OpenAICompatibleJudge):
    """OpenRouter judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or "
                "pass api_key= to OpenRouterJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )


class XAIJudge(OpenAICompatibleJudge):
    """X.AI / Grok judge using their OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "grok-2-latest",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("XAI_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "X.AI API key not found. Set XAI_API_KEY or pass api_key= to XAIJudge()."
            )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url="https://api.x.ai/v1",
            **kwargs,
        )


class CohereJudge:
    """Cohere-based LLM judge using the ``cohere`` SDK."""

    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Cohere API key not found. Set COHERE_API_KEY or pass api_key= to CohereJudge()."
            )

        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "cohere package not installed. Install it with:\n  pip install cohere"
            )

        self._client = cohere.AsyncClientV2(api_key=resolved_key)

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Evaluate using the Cohere chat endpoint."""
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

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        raw_output = ""
        if response.message and response.message.content:
            raw_output = response.message.content[0].text or ""

        cost = 0.0
        usage = getattr(response, "usage", None)
        if usage:
            billed_input = getattr(usage, "billed_units", None)
            if billed_input:
                input_tokens = getattr(billed_input, "input_tokens", 0) or 0
                output_tokens = getattr(billed_input, "output_tokens", 0) or 0
                cost = estimate_cost(self.model, input_tokens, output_tokens)
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
        return f"CohereJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


class MistralJudge:
    """Mistral AI judge using the ``mistralai`` SDK."""

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise JudgeConfigError(
                "Mistral API key not found. Set MISTRAL_API_KEY or pass api_key= to MistralJudge()."
            )

        try:
            from mistralai import Mistral  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "mistralai package not installed. Install it with:\n  pip install mistralai"
            )

        self._client = Mistral(api_key=resolved_key)

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Evaluate using the Mistral chat completions endpoint."""
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

        response = await self._client.chat.complete_async(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        raw_output = ""
        if response and response.choices:
            raw_output = response.choices[0].message.content or ""

        cost = 0.0
        if response and response.usage:
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
        return f"MistralJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


class BedrockJudge:
    """AWS Bedrock judge using ``boto3``."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            raise JudgeConfigError(
                "boto3 package not installed. Install it with:\n  pip install boto3"
            )

        resolved_region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=resolved_region)

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Evaluate using AWS Bedrock's invoke-model API.

        Note: boto3 is synchronous, so we run the call in the current thread
        to keep the interface async-compatible.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._invoke, prompt, system_prompt)

    def _invoke(self, prompt: str, system_prompt: str | None) -> JudgeResponse:
        """Synchronous Bedrock invocation."""
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
            messages.append({"role": "assistant", "content": "Understood."})
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

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": messages,
        }

        trace_headers = propagate_trace_context()

        def _inject_headers(request: Any, **_kwargs: Any) -> None:
            for key, value in trace_headers.items():
                request.headers[key] = value

        events = getattr(self._client.meta, "events", None)
        if events is not None and trace_headers:
            events.register_first("before-sign.bedrock-runtime.*", _inject_headers)
        try:
            response = self._client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        finally:
            if events is not None and trace_headers:
                events.unregister("before-sign.bedrock-runtime.*", _inject_headers)

        response_body = json.loads(response["body"].read())
        raw_output = ""
        if response_body.get("content"):
            raw_output = response_body["content"][0].get("text", "")

        cost = 0.0
        usage = response_body.get("usage")
        if usage:
            cost = estimate_cost(
                self.model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
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
        return f"BedrockJudge(model={self.model!r}, total_cost=${self.total_cost:.4f})"


# ---------------------------------------------------------------------------
# VertexAIJudge
# ---------------------------------------------------------------------------


class VertexAIJudge:
    """Google Vertex AI judge using the ``google-cloud-aiplatform`` SDK.

    Talks directly to Vertex AI's Gemini endpoints so GCP enterprises do
    not have to proxy through LiteLLM or AI Studio. Credentials are
    resolved in the standard Application Default Credentials (ADC) way
    when ``credentials`` is ``None``.

    Args:
        model: A Vertex AI Gemini model ID, e.g. ``"gemini-1.5-pro"``,
            ``"gemini-1.5-flash"``, or ``"gemini-2.0-flash-exp"``.
        project: GCP project ID. Falls back to ``GOOGLE_CLOUD_PROJECT``
            or ``GCP_PROJECT`` from the environment.
        location: Vertex AI region, e.g. ``"us-central1"``. Falls back
            to ``GOOGLE_CLOUD_LOCATION`` / ``GCP_LOCATION`` or
            ``"us-central1"``.
        credentials: Optional ``google.auth.credentials.Credentials``
            instance. When ``None`` (the default) ADC is used.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        project: str | None = None,
        location: str | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.model = model
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

        resolved_project = (
            project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        )
        if not resolved_project:
            raise JudgeConfigError(
                "Vertex AI project not found. Set the GOOGLE_CLOUD_PROJECT "
                "environment variable or pass project= to VertexAIJudge()."
            )

        resolved_location = (
            location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GCP_LOCATION")
            or "us-central1"
        )
        self._project = resolved_project
        self._location = resolved_location

        try:
            import vertexai  # type: ignore[import-untyped]
            from vertexai.generative_models import (  # type: ignore[import-untyped]
                GenerativeModel,
            )
        except ImportError:
            raise JudgeConfigError(
                "google-cloud-aiplatform package not installed. Install it with:\n"
                "  pip install checkllm[vertex]"
            )

        init_kwargs: dict[str, Any] = {
            "project": resolved_project,
            "location": resolved_location,
        }
        if credentials is not None:
            init_kwargs["credentials"] = credentials

        vertexai.init(**init_kwargs)
        self._vertexai = vertexai
        self._gmodel = GenerativeModel(model)

    @_transient_retry
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Evaluate using the Vertex AI Gemini endpoint.

        Args:
            prompt: The user turn sent to the model.
            system_prompt: Optional preamble prepended to ``prompt``.

        Returns:
            A parsed :class:`JudgeResponse` including cost when the
            response carries usage metadata.
        """
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

        raw_output = ""
        text_attr = getattr(response, "text", None)
        if text_attr:
            raw_output = text_attr
        else:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                content = getattr(candidates[0], "content", None)
                parts = getattr(content, "parts", None) or []
                if parts:
                    raw_output = getattr(parts[0], "text", "") or ""

        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            cost = _gemini_estimate_cost(self.model, prompt_tokens, completion_tokens)
        self.last_cost = cost
        self.total_cost += cost
        self.last_input_tokens = prompt_tokens
        self.last_output_tokens = completion_tokens

        score, reasoning = _parse_judge_json(raw_output)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=raw_output,
            cost=cost,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=self.model,
            provider="vertex",
        )

    def __repr__(self) -> str:
        return (
            f"VertexAIJudge(model={self.model!r}, project={self._project!r}, "
            f"location={self._location!r}, total_cost=${self.total_cost:.4f})"
        )


# Factory
# ---------------------------------------------------------------------------


def create_judge(backend: str, **kwargs: Any) -> JudgeBackend:
    """Create a judge backend by name.

    Parameters
    ----------
    backend:
        One of "openai", "anthropic", "gemini", "vertex", "azure",
        "ollama", "litellm", "custom", "cohere", "mistral", "deepseek",
        "groq", "together", "fireworks", "perplexity", "vllm",
        "bedrock", "openrouter", or "xai".

        Model-style aliases are also recognized — e.g. ``"deepseek-chat"``
        or ``"deepseek-reasoner"`` resolve to ``DeepSeekJudge`` with
        ``model=<alias>``.
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
        "vertex": VertexAIJudge,
        "vertexai": VertexAIJudge,
        "azure": AzureOpenAIJudge,
        "ollama": OllamaJudge,
        "litellm": LiteLLMJudge,
        "custom": CustomHTTPJudge,
        "cohere": CohereJudge,
        "mistral": MistralJudge,
        "deepseek": DeepSeekJudge,
        "groq": GroqJudge,
        "together": TogetherJudge,
        "fireworks": FireworksJudge,
        "perplexity": PerplexityJudge,
        "vllm": VLLMJudge,
        "bedrock": BedrockJudge,
        "openrouter": OpenRouterJudge,
        "xai": XAIJudge,
    }

    # Model-style aliases — e.g. "deepseek-chat" / "deepseek-reasoner"
    # route to the deepseek backend with the model baked in.
    if backend in {"deepseek-chat", "deepseek-reasoner"}:
        kwargs.setdefault("model", backend)
        return DeepSeekJudge(**kwargs)  # type: ignore[return-value]

    cls = _backends.get(backend)
    if cls is None:
        supported = ", ".join(sorted(_backends))
        raise ValueError(f"Unknown backend {backend!r}. Supported: {supported}")

    return cls(**kwargs)  # type: ignore[return-value]
