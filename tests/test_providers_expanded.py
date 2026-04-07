"""Tests for expanded provider backends in checkllm.providers.

Covers OpenAICompatibleJudge base class, all new provider judges
(DeepSeek, Groq, Together, Fireworks, Perplexity, vLLM, OpenRouter,
X.AI, Cohere, Mistral, Bedrock), and factory/discovery integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.judge import JudgeBackend, JudgeConfigError
from checkllm.models import JudgeResponse


def _make_httpx_chat_response(
    score: float = 0.85,
    reasoning: str = "Good output",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a mock httpx response mimicking an OpenAI chat/completions reply."""
    body = {
        "choices": [
            {"message": {"content": json.dumps({"score": score, "reasoning": reasoning})}}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_httpx_client(mock_resp: MagicMock) -> MagicMock:
    """Wrap a mock response in an async httpx client context manager."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client
    return mock_httpx


class TestOpenAICompatibleJudge:
    """Tests for the shared OpenAI-compatible base class."""

    def test_constructor_stores_fields(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="test-model",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
        )
        assert judge.model == "test-model"
        assert judge._api_key == "sk-test"
        assert judge._base_url == "https://api.example.com/v1"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    def test_base_url_trailing_slash_stripped(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="m", api_key="k", base_url="https://api.example.com/v1/"
        )
        assert judge._base_url == "https://api.example.com/v1"

    @pytest.mark.asyncio
    async def test_evaluate_sends_correct_request(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="test-model",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
        )

        mock_resp = _make_httpx_chat_response(0.9, "Great")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("Rate this output", system_prompt="Be strict")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.9
        assert result.reasoning == "Great"

        call_args = mock_httpx.AsyncClient.return_value.post.call_args
        assert "chat/completions" in call_args[0][0]
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test"
        payload = call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.0
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_evaluate_without_system_prompt(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="m", api_key="k", base_url="https://api.example.com/v1"
        )

        mock_resp = _make_httpx_chat_response()
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            await judge.evaluate("test prompt")

        call_args = mock_httpx.AsyncClient.return_value.post.call_args
        payload = call_args[1]["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_evaluate_no_api_key_omits_auth_header(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="m", api_key=None, base_url="http://localhost:8000/v1"
        )

        mock_resp = _make_httpx_chat_response()
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            await judge.evaluate("test")

        headers = mock_httpx.AsyncClient.return_value.post.call_args[1]["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_evaluate_tracks_cost(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="gpt-4o", api_key="k", base_url="https://api.example.com/v1"
        )
        mock_resp = _make_httpx_chat_response(prompt_tokens=1000, completion_tokens=500)
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.cost > 0.0
        assert judge.total_cost > 0.0
        assert judge.last_cost == result.cost

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="m", api_key="k", base_url="https://api.example.com/v1"
        )

        body = {
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    def test_repr_includes_class_name(self) -> None:
        from checkllm.providers import OpenAICompatibleJudge

        judge = OpenAICompatibleJudge(
            model="test", api_key="k", base_url="https://example.com/v1"
        )
        r = repr(judge)
        assert "OpenAICompatibleJudge" in r
        assert "test" in r
        assert "example.com" in r


class TestDeepSeekJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from checkllm.providers import DeepSeekJudge

        with pytest.raises(JudgeConfigError, match="DEEPSEEK_API_KEY"):
            DeepSeekJudge()

    def test_constructor_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
        from checkllm.providers import DeepSeekJudge

        judge = DeepSeekJudge()
        assert judge.model == "deepseek-chat"
        assert judge._base_url == "https://api.deepseek.com/v1"
        assert isinstance(judge, JudgeBackend)

    def test_constructor_with_explicit_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from checkllm.providers import DeepSeekJudge

        judge = DeepSeekJudge(api_key="ds-explicit")
        assert judge._api_key == "ds-explicit"

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
        from checkllm.providers import DeepSeekJudge

        judge = DeepSeekJudge()
        mock_resp = _make_httpx_chat_response(0.8, "Solid")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.8
        assert result.reasoning == "Solid"


class TestGroqJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from checkllm.providers import GroqJudge

        with pytest.raises(JudgeConfigError, match="GROQ_API_KEY"):
            GroqJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from checkllm.providers import GroqJudge

        judge = GroqJudge()
        assert judge.model == "llama-3.3-70b-versatile"
        assert "groq.com" in judge._base_url
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from checkllm.providers import GroqJudge

        judge = GroqJudge()
        mock_resp = _make_httpx_chat_response(0.7, "OK")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.7


class TestTogetherJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        from checkllm.providers import TogetherJudge

        with pytest.raises(JudgeConfigError, match="TOGETHER_API_KEY"):
            TogetherJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOGETHER_API_KEY", "tog-test")
        from checkllm.providers import TogetherJudge

        judge = TogetherJudge()
        assert "Meta-Llama" in judge.model
        assert "together.xyz" in judge._base_url
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOGETHER_API_KEY", "tog-test")
        from checkllm.providers import TogetherJudge

        judge = TogetherJudge()
        mock_resp = _make_httpx_chat_response(0.65, "Decent")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.65


class TestFireworksJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        from checkllm.providers import FireworksJudge

        with pytest.raises(JudgeConfigError, match="FIREWORKS_API_KEY"):
            FireworksJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
        from checkllm.providers import FireworksJudge

        judge = FireworksJudge()
        assert "fireworks" in judge.model
        assert "fireworks.ai" in judge._base_url
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
        from checkllm.providers import FireworksJudge

        judge = FireworksJudge()
        mock_resp = _make_httpx_chat_response(0.95, "Excellent")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.95


class TestPerplexityJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        from checkllm.providers import PerplexityJudge

        with pytest.raises(JudgeConfigError, match="PERPLEXITY_API_KEY"):
            PerplexityJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
        from checkllm.providers import PerplexityJudge

        judge = PerplexityJudge()
        assert "sonar" in judge.model
        assert "perplexity.ai" in judge._base_url
        assert isinstance(judge, JudgeBackend)


class TestVLLMJudge:
    def test_constructor_defaults(self) -> None:
        from checkllm.providers import VLLMJudge

        judge = VLLMJudge()
        assert judge.model == "default"
        assert judge._base_url == "http://localhost:8000/v1"
        assert judge._api_key is None
        assert isinstance(judge, JudgeBackend)

    def test_constructor_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-server:9000/v1")
        from checkllm.providers import VLLMJudge

        judge = VLLMJudge()
        assert judge._base_url == "http://gpu-server:9000/v1"

    def test_constructor_explicit_base_url(self) -> None:
        from checkllm.providers import VLLMJudge

        judge = VLLMJudge(base_url="http://custom:5000/v1")
        assert judge._base_url == "http://custom:5000/v1"

    @pytest.mark.asyncio
    async def test_evaluate(self) -> None:
        from checkllm.providers import VLLMJudge

        judge = VLLMJudge(model="my-model")
        mock_resp = _make_httpx_chat_response(0.75, "Reasonable")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.75


class TestOpenRouterJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from checkllm.providers import OpenRouterJudge

        with pytest.raises(JudgeConfigError, match="OPENROUTER_API_KEY"):
            OpenRouterJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        from checkllm.providers import OpenRouterJudge

        judge = OpenRouterJudge()
        assert "claude" in judge.model
        assert "openrouter.ai" in judge._base_url
        assert isinstance(judge, JudgeBackend)


class TestXAIJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from checkllm.providers import XAIJudge

        with pytest.raises(JudgeConfigError, match="XAI_API_KEY"):
            XAIJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        from checkllm.providers import XAIJudge

        judge = XAIJudge()
        assert judge.model == "grok-2-latest"
        assert "x.ai" in judge._base_url
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        from checkllm.providers import XAIJudge

        judge = XAIJudge()
        mock_resp = _make_httpx_chat_response(0.88, "Well done")
        mock_httpx = _mock_httpx_client(mock_resp)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await judge.evaluate("test")

        assert result.score == 0.88


class TestCohereJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from checkllm.providers import CohereJudge

        with pytest.raises(JudgeConfigError, match="COHERE_API_KEY"):
            CohereJudge()

    def test_missing_package_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        import builtins

        real_import = builtins.__import__

        def _block_cohere(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cohere" or name.startswith("cohere."):
                raise ImportError("no cohere")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_cohere):
            from checkllm.providers import CohereJudge

            with pytest.raises(JudgeConfigError, match="cohere"):
                CohereJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from checkllm.providers import CohereJudge

            judge = CohereJudge()
        assert judge.model == "command-r-plus"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        mock_cohere = MagicMock()

        mock_text_block = MagicMock()
        mock_text_block.text = '{"score": 0.82, "reasoning": "Clear answer"}'

        mock_msg = MagicMock()
        mock_msg.content = [mock_text_block]

        mock_response = MagicMock()
        mock_response.message = mock_msg
        mock_response.usage = None

        mock_client_instance = MagicMock()
        mock_client_instance.chat = AsyncMock(return_value=mock_response)
        mock_cohere.AsyncClientV2.return_value = mock_client_instance

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from checkllm.providers import CohereJudge

            judge = CohereJudge()
            result = await judge.evaluate("test", system_prompt="Be fair")

        assert result.score == 0.82
        assert result.reasoning == "Clear answer"

    def test_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from checkllm.providers import CohereJudge

            judge = CohereJudge()
        assert "CohereJudge" in repr(judge)
        assert "command-r-plus" in repr(judge)


class TestMistralJudge:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from checkllm.providers import MistralJudge

        with pytest.raises(JudgeConfigError, match="MISTRAL_API_KEY"):
            MistralJudge()

    def test_missing_package_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        import builtins

        real_import = builtins.__import__

        def _block_mistral(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mistralai" or name.startswith("mistralai."):
                raise ImportError("no mistralai")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_mistral):
            from checkllm.providers import MistralJudge

            with pytest.raises(JudgeConfigError, match="mistralai"):
                MistralJudge()

    def test_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        mock_mistral = MagicMock()
        with patch.dict("sys.modules", {"mistralai": mock_mistral}):
            from checkllm.providers import MistralJudge

            judge = MistralJudge()
        assert judge.model == "mistral-large-latest"
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        mock_mistral_mod = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.77, "reasoning": "Accurate"}'

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 40

        mock_client_instance = MagicMock()
        mock_client_instance.chat.complete_async = AsyncMock(return_value=mock_response)
        mock_mistral_mod.Mistral.return_value = mock_client_instance

        with patch.dict("sys.modules", {"mistralai": mock_mistral_mod}):
            from checkllm.providers import MistralJudge

            judge = MistralJudge()
            result = await judge.evaluate("test", system_prompt="Judge carefully")

        assert result.score == 0.77
        assert result.reasoning == "Accurate"

    def test_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        mock_mistral = MagicMock()
        with patch.dict("sys.modules", {"mistralai": mock_mistral}):
            from checkllm.providers import MistralJudge

            judge = MistralJudge()
        assert "MistralJudge" in repr(judge)
        assert "mistral-large-latest" in repr(judge)


class TestBedrockJudge:
    def test_missing_package_raises(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_boto3(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "boto3" or name.startswith("boto3."):
                raise ImportError("no boto3")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_boto3):
            from checkllm.providers import BedrockJudge

            with pytest.raises(JudgeConfigError, match="boto3"):
                BedrockJudge()

    def test_constructor(self) -> None:
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from checkllm.providers import BedrockJudge

            judge = BedrockJudge()
        assert "claude-3-sonnet" in judge.model
        assert judge.total_cost == 0.0
        assert isinstance(judge, JudgeBackend)

    def test_constructor_custom_region(self) -> None:
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from checkllm.providers import BedrockJudge

            judge = BedrockJudge(region="eu-west-1")
        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime", region_name="eu-west-1"
        )
        assert isinstance(judge, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate(self) -> None:
        mock_boto3 = MagicMock()

        response_body = {
            "content": [{"text": '{"score": 0.92, "reasoning": "Strong"}'}],
            "usage": {"input_tokens": 120, "output_tokens": 60},
        }
        mock_body_stream = MagicMock()
        mock_body_stream.read.return_value = json.dumps(response_body).encode()

        mock_boto3.client.return_value.invoke_model.return_value = {
            "body": mock_body_stream,
        }

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from checkllm.providers import BedrockJudge

            judge = BedrockJudge()
            result = await judge.evaluate("test prompt", system_prompt="Be thorough")

        assert isinstance(result, JudgeResponse)
        assert result.score == 0.92
        assert result.reasoning == "Strong"

    def test_repr(self) -> None:
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from checkllm.providers import BedrockJudge

            judge = BedrockJudge()
        r = repr(judge)
        assert "BedrockJudge" in r
        assert "claude-3-sonnet" in r


class TestFactoryNewBackends:
    """Verify create_judge() recognizes all new backends."""

    def test_deepseek_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
        from checkllm.providers import create_judge

        judge = create_judge("deepseek")
        assert judge.__class__.__name__ == "DeepSeekJudge"

    def test_groq_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from checkllm.providers import create_judge

        judge = create_judge("groq")
        assert judge.__class__.__name__ == "GroqJudge"

    def test_together_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOGETHER_API_KEY", "tog-test")
        from checkllm.providers import create_judge

        judge = create_judge("together")
        assert judge.__class__.__name__ == "TogetherJudge"

    def test_fireworks_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
        from checkllm.providers import create_judge

        judge = create_judge("fireworks")
        assert judge.__class__.__name__ == "FireworksJudge"

    def test_perplexity_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
        from checkllm.providers import create_judge

        judge = create_judge("perplexity")
        assert judge.__class__.__name__ == "PerplexityJudge"

    def test_vllm_backend(self) -> None:
        from checkllm.providers import create_judge

        judge = create_judge("vllm", model="my-local-model")
        assert judge.__class__.__name__ == "VLLMJudge"

    def test_openrouter_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        from checkllm.providers import create_judge

        judge = create_judge("openrouter")
        assert judge.__class__.__name__ == "OpenRouterJudge"

    def test_xai_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        from checkllm.providers import create_judge

        judge = create_judge("xai")
        assert judge.__class__.__name__ == "XAIJudge"

    def test_cohere_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from checkllm.providers import create_judge

            judge = create_judge("cohere")
        assert judge.__class__.__name__ == "CohereJudge"

    def test_mistral_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        mock_mistral = MagicMock()
        with patch.dict("sys.modules", {"mistralai": mock_mistral}):
            from checkllm.providers import create_judge

            judge = create_judge("mistral")
        assert judge.__class__.__name__ == "MistralJudge"

    def test_bedrock_backend(self) -> None:
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from checkllm.providers import create_judge

            judge = create_judge("bedrock")
        assert judge.__class__.__name__ == "BedrockJudge"

    def test_all_backends_registered(self) -> None:
        """Ensure every expected backend name is accepted by the factory."""
        expected = {
            "openai", "anthropic", "gemini", "azure", "ollama", "litellm",
            "custom", "cohere", "mistral", "deepseek", "groq", "together",
            "fireworks", "perplexity", "vllm", "bedrock", "openrouter", "xai",
        }
        from checkllm.providers import create_judge

        for name in expected:
            try:
                create_judge(name)
            except (JudgeConfigError, ValueError):
                # JudgeConfigError = missing key/pkg, expected
                # ValueError = unknown backend, should NOT happen
                pass
            except Exception:
                pass

        # The key test: none should raise ValueError
        from checkllm.providers import create_judge as cj

        for name in expected:
            try:
                cj(name)
            except ValueError:
                pytest.fail(f"Backend {name!r} not registered in create_judge()")
            except Exception:
                pass  # Config errors are fine


class TestDiscoveryNewProviders:
    """Verify auto-detection finds new providers from env vars."""

    def test_detect_cohere(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"COHERE_API_KEY": "co-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "cohere"
        assert result[1] == "command-r-plus"

    def test_detect_mistral(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"MISTRAL_API_KEY": "mist-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "mistral"

    def test_detect_deepseek(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "ds-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "deepseek"

    def test_detect_groq(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "groq"

    def test_detect_together(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"TOGETHER_API_KEY": "tog-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "together"

    def test_detect_fireworks(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"FIREWORKS_API_KEY": "fw-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "fireworks"

    def test_detect_perplexity(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pplx-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "perplexity"

    def test_detect_openrouter(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "openrouter"

    def test_detect_xai(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"XAI_API_KEY": "xai-test"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "xai"

    def test_detect_bedrock_via_access_key(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict(
            "os.environ", {"AWS_ACCESS_KEY_ID": "AKIA-test"}, clear=True
        ):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "bedrock"

    def test_detect_bedrock_via_profile(self) -> None:
        from checkllm.discovery import detect_judge_backend

        with patch.dict("os.environ", {"AWS_PROFILE": "dev"}, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "bedrock"

    def test_openai_still_wins_priority(self) -> None:
        """OpenAI should take priority even when many keys are present."""
        from checkllm.discovery import detect_judge_backend

        env = {
            "OPENAI_API_KEY": "sk-test",
            "COHERE_API_KEY": "co-test",
            "GROQ_API_KEY": "gsk-test",
            "DEEPSEEK_API_KEY": "ds-test",
        }
        with patch.dict("os.environ", env, clear=True):
            result = detect_judge_backend()
        assert result is not None
        assert result[0] == "openai"
