"""Tests for checkllm.providers – multi-provider judge backends."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.judge import JudgeBackend, JudgeConfigError
from checkllm.models import JudgeResponse


# ===================================================================
# GeminiJudge
# ===================================================================


class TestGeminiJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from checkllm.providers import GeminiJudge

        with pytest.raises(JudgeConfigError, match="GOOGLE_API_KEY"):
            GeminiJudge(model="gemini-2.0-flash")

    def test_missing_package_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import builtins

        real_import = builtins.__import__

        def _block_genai(name: str, *args: Any, **kwargs: Any) -> Any:
            if "google.generativeai" in name or name == "google":
                raise ImportError("no google")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_genai):
            from checkllm.providers import GeminiJudge

            with pytest.raises(JudgeConfigError, match="google-generativeai"):
                GeminiJudge(model="gemini-2.0-flash")

    def test_constructor_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            from checkllm.providers import GeminiJudge

            judge = GeminiJudge(model="gemini-2.0-flash", api_key="test-key")
        assert judge.model == "gemini-2.0-flash"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_genai = MagicMock()

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50

        mock_response = MagicMock()
        mock_response.text = '{"score": 0.9, "reasoning": "Good answer"}'
        mock_response.usage_metadata = mock_usage

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            from checkllm.providers import GeminiJudge

            judge = GeminiJudge(model="gemini-2.0-flash", api_key="test-key")

        # Directly patch the model's async method on the stored instance
        judge._gmodel.generate_content_async = AsyncMock(return_value=mock_response)

        result = await judge.evaluate("Rate this output", system_prompt="Be a judge")
        assert isinstance(result, JudgeResponse)
        assert result.score == 0.9
        assert "Good answer" in result.reasoning
        assert judge.total_cost > 0.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_genai = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Not JSON at all"
        mock_response.usage_metadata = None

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            from checkllm.providers import GeminiJudge

            judge = GeminiJudge(model="gemini-2.0-flash", api_key="test-key")

        judge._gmodel.generate_content_async = AsyncMock(return_value=mock_response)

        result = await judge.evaluate("Rate this")
        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    def test_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            from checkllm.providers import GeminiJudge

            judge = GeminiJudge(model="gemini-2.0-flash", api_key="test-key")
        assert "GeminiJudge" in repr(judge)
        assert "gemini-2.0-flash" in repr(judge)


# ===================================================================
# AzureOpenAIJudge
# ===================================================================


class TestAzureOpenAIJudge:
    @pytest.fixture(autouse=True)
    def _clear_azure_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    def test_missing_api_key_raises(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        with pytest.raises(JudgeConfigError, match="AZURE_OPENAI_API_KEY"):
            AzureOpenAIJudge()

    def test_missing_endpoint_raises(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        with pytest.raises(JudgeConfigError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAIJudge(api_key="test-key")

    def test_missing_deployment_raises(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        with pytest.raises(JudgeConfigError, match="AZURE_OPENAI_DEPLOYMENT"):
            AzureOpenAIJudge(api_key="test-key", endpoint="https://my.openai.azure.com")

    def test_constructor(self) -> None:
        mock_azure_client = MagicMock()
        with patch("checkllm.providers.AsyncAzureOpenAI", mock_azure_client, create=True):
            with patch.dict("sys.modules", {}):
                from checkllm.providers import AzureOpenAIJudge

                judge = AzureOpenAIJudge(
                    api_key="test-key",
                    endpoint="https://my.openai.azure.com",
                    deployment="my-gpt4",
                )
        assert judge.model == "gpt-4o"
        assert judge._deployment == "my-gpt4"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate_parses_response(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.75, "reasoning": "Decent"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100

        judge = AzureOpenAIJudge(
            api_key="test-key",
            endpoint="https://my.openai.azure.com",
            deployment="my-gpt4",
        )
        with patch.object(
            judge._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await judge.evaluate("test prompt", system_prompt="Be strict")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.75
        assert result.reasoning == "Decent"

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        mock_choice = MagicMock()
        mock_choice.message.content = "BROKEN"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        judge = AzureOpenAIJudge(
            api_key="test-key",
            endpoint="https://my.openai.azure.com",
            deployment="my-gpt4",
        )
        with patch.object(
            judge._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await judge.evaluate("test")

        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    def test_repr(self) -> None:
        from checkllm.providers import AzureOpenAIJudge

        judge = AzureOpenAIJudge(
            api_key="test-key",
            endpoint="https://my.openai.azure.com",
            deployment="my-gpt4",
        )
        r = repr(judge)
        assert "AzureOpenAIJudge" in r
        assert "my-gpt4" in r


# ===================================================================
# OllamaJudge
# ===================================================================


class TestOllamaJudge:
    def test_constructor_defaults(self) -> None:
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge()
        assert judge.model == "llama3.1"
        assert judge._host == "http://localhost:11434"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    def test_constructor_env_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://myserver:11434")
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge()
        assert judge._host == "http://myserver:11434"

    def test_constructor_explicit_host(self) -> None:
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge(host="http://custom:1234")
        assert judge._host == "http://custom:1234"

    @pytest.mark.asyncio
    async def test_evaluate_httpx(self) -> None:
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge(model="llama3.1")

        response_data = {"response": '{"score": 0.8, "reasoning": "Solid answer"}'}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("Rate this", system_prompt="Be fair")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.8
        assert result.reasoning == "Solid answer"
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_malformed_json(self) -> None:
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge(model="llama3.1")

        response_data = {"response": "not json at all"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("Rate this")

        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    def test_repr(self) -> None:
        from checkllm.providers import OllamaJudge

        judge = OllamaJudge(model="mixtral", host="http://localhost:11434")
        r = repr(judge)
        assert "OllamaJudge" in r
        assert "mixtral" in r


# ===================================================================
# LiteLLMJudge
# ===================================================================


class TestLiteLLMJudge:
    def test_missing_package_raises(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_litellm(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "litellm" or name.startswith("litellm."):
                raise ImportError("no litellm")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_litellm):
            from checkllm.providers import LiteLLMJudge

            with pytest.raises(JudgeConfigError, match="litellm"):
                LiteLLMJudge(model="gpt-4o")

    def test_constructor(self) -> None:
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from checkllm.providers import LiteLLMJudge

            judge = LiteLLMJudge(model="gpt-4o", api_key="test-key")
        assert judge.model == "gpt-4o"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate_parses_response(self) -> None:
        mock_litellm = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.95, "reasoning": "Excellent"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 75

        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        mock_litellm.completion_cost.return_value = 0.005

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from checkllm.providers import LiteLLMJudge

            judge = LiteLLMJudge(model="gpt-4o")
            result = await judge.evaluate("test prompt", system_prompt="You judge")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.95
        assert result.reasoning == "Excellent"
        assert result.cost == 0.005

    @pytest.mark.asyncio
    async def test_evaluate_cost_fallback(self) -> None:
        """When litellm.completion_cost raises, fall back to estimate_cost."""
        mock_litellm = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.5, "reasoning": "OK"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        mock_litellm.completion_cost.side_effect = Exception("no cost data")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from checkllm.providers import LiteLLMJudge

            judge = LiteLLMJudge(model="gpt-4o")
            result = await judge.evaluate("test")

        assert result.score == 0.5
        assert result.cost > 0.0  # should fall back to estimate_cost

    def test_repr(self) -> None:
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from checkllm.providers import LiteLLMJudge

            judge = LiteLLMJudge(model="claude-sonnet-4-6")
        assert "LiteLLMJudge" in repr(judge)
        assert "claude-sonnet-4-6" in repr(judge)


# ===================================================================
# CustomHTTPJudge
# ===================================================================


class TestCustomHTTPJudge:
    def test_constructor(self) -> None:
        from checkllm.providers import CustomHTTPJudge

        judge = CustomHTTPJudge(url="http://example.com/judge")
        assert judge.model == "custom:http://example.com/judge"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate_default_parser(self) -> None:
        from checkllm.providers import CustomHTTPJudge

        judge = CustomHTTPJudge(
            url="http://example.com/judge",
            headers={"Authorization": "Bearer tok123"},
        )

        response_data = {"score": 0.7, "reasoning": "Pretty good"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.TimeoutException = TimeoutError
        mock_httpx.ConnectError = ConnectionError

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test prompt", system_prompt="strict judge")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.7
        assert result.reasoning == "Pretty good"

    @pytest.mark.asyncio
    async def test_evaluate_custom_parser(self) -> None:
        from checkllm.providers import CustomHTTPJudge

        def my_parser(data: dict[str, Any]) -> tuple[float, str]:
            return float(data["result"]["value"]), data["result"]["explanation"]

        judge = CustomHTTPJudge(
            url="http://example.com/api",
            response_parser=my_parser,
        )

        response_data = {"result": {"value": 0.6, "explanation": "Decent"}}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.TimeoutException = TimeoutError
        mock_httpx.ConnectError = ConnectionError

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.6
        assert result.reasoning == "Decent"

    @pytest.mark.asyncio
    async def test_evaluate_broken_parser(self) -> None:
        """When the custom parser raises, score should be 0.0."""
        from checkllm.providers import CustomHTTPJudge

        def bad_parser(data: dict[str, Any]) -> tuple[float, str]:
            raise KeyError("missing")

        judge = CustomHTTPJudge(
            url="http://example.com/api",
            response_parser=bad_parser,
        )

        response_data = {"unexpected": "structure"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.TimeoutException = TimeoutError
        mock_httpx.ConnectError = ConnectionError

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_missing_httpx_raises(self) -> None:
        from checkllm.providers import CustomHTTPJudge

        judge = CustomHTTPJudge(url="http://example.com/api")

        import builtins

        real_import = builtins.__import__

        def _block_httpx(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "httpx":
                raise ImportError("no httpx")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_httpx):
            with pytest.raises(JudgeConfigError, match="httpx"):
                await judge.evaluate("test")

    def test_repr(self) -> None:
        from checkllm.providers import CustomHTTPJudge

        judge = CustomHTTPJudge(url="http://example.com/judge")
        assert "CustomHTTPJudge" in repr(judge)
        assert "http://example.com/judge" in repr(judge)


# ===================================================================
# create_judge factory
# ===================================================================


class TestCreateJudge:
    def test_openai_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from checkllm.providers import create_judge

        judge = create_judge("openai")
        assert judge.__class__.__name__ == "OpenAIJudge"

    def test_anthropic_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from checkllm.providers import create_judge

            judge = create_judge("anthropic")
        assert judge.__class__.__name__ == "AnthropicJudge"

    def test_gemini_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            from checkllm.providers import create_judge

            judge = create_judge("gemini")
        assert judge.__class__.__name__ == "GeminiJudge"

    def test_azure_backend(self) -> None:
        from checkllm.providers import create_judge

        judge = create_judge(
            "azure",
            api_key="test-key",
            endpoint="https://my.openai.azure.com",
            deployment="my-gpt4",
        )
        assert judge.__class__.__name__ == "AzureOpenAIJudge"

    def test_ollama_backend(self) -> None:
        from checkllm.providers import create_judge

        judge = create_judge("ollama", model="mixtral")
        assert judge.__class__.__name__ == "OllamaJudge"

    def test_litellm_backend(self) -> None:
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from checkllm.providers import create_judge

            judge = create_judge("litellm", model="gpt-4o")
        assert judge.__class__.__name__ == "LiteLLMJudge"

    def test_custom_backend(self) -> None:
        from checkllm.providers import create_judge

        judge = create_judge("custom", url="http://example.com/judge")
        assert judge.__class__.__name__ == "CustomHTTPJudge"

    def test_unknown_backend_raises(self) -> None:
        from checkllm.providers import create_judge

        with pytest.raises(ValueError, match="Unknown backend"):
            create_judge("nonexistent")


# ===================================================================
# Shared helpers
# ===================================================================


class TestParseJudgeJson:
    def test_valid_json(self) -> None:
        from checkllm.providers import _parse_judge_json

        score, reasoning = _parse_judge_json('{"score": 0.5, "reasoning": "OK"}')
        assert score == 0.5
        assert reasoning == "OK"

    def test_score_clamped_high(self) -> None:
        from checkllm.providers import _parse_judge_json

        score, _ = _parse_judge_json('{"score": 1.5, "reasoning": "too high"}')
        assert score == 1.0

    def test_score_clamped_low(self) -> None:
        from checkllm.providers import _parse_judge_json

        score, _ = _parse_judge_json('{"score": -0.5, "reasoning": "too low"}')
        assert score == 0.0

    def test_invalid_json(self) -> None:
        from checkllm.providers import _parse_judge_json

        score, reasoning = _parse_judge_json("not json")
        assert score == 0.0
        assert "failed" in reasoning.lower() or "parse" in reasoning.lower()

    def test_missing_fields(self) -> None:
        from checkllm.providers import _parse_judge_json

        score, reasoning = _parse_judge_json("{}")
        assert score == 0.0
        assert reasoning == ""
