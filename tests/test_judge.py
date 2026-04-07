from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.judge import JudgeBackend, JudgeConfigError, OpenAIJudge, estimate_cost
from checkllm.models import JudgeResponse


class TestJudgeBackendProtocol:
    def test_openai_judge_satisfies_protocol(self):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        assert isinstance(judge, JudgeBackend)


class TestOpenAIJudge:
    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI chat completion response."""
        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.85, "reasoning": "Output is well grounded"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        return mock_response

    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_response(self, mock_openai_response):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            response = await judge.evaluate(
                "Rate this output for hallucination: 'The sky is blue'"
            )
        assert isinstance(response, JudgeResponse)
        assert response.score == 0.85
        assert "grounded" in response.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_system_prompt(self, mock_openai_response):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            await judge.evaluate("test prompt", system_prompt="You are a judge.")
            call_args = mock_create.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a judge."

    @pytest.mark.asyncio
    async def test_evaluate_tracks_cost(self, mock_openai_response):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            response = await judge.evaluate("test prompt")
        assert response.score >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(self):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        mock_choice = MagicMock()
        mock_choice.message.content = "This is not JSON at all"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            response = await judge.evaluate("test prompt")
        assert response.score == 0.0
        assert "parse" in response.reasoning.lower() or "failed" in response.reasoning.lower()


class TestJudgeConfigError:
    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(JudgeConfigError, match="OPENAI_API_KEY"):
            OpenAIJudge(model="gpt-4o")

    def test_cost_tracking(self):
        judge = OpenAIJudge(model="gpt-4o", api_key="test-key")
        assert judge.total_cost == 0.0
        assert repr(judge) == "OpenAIJudge(model='gpt-4o', total_cost=$0.0000)"


class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_uses_default(self):
        cost = estimate_cost("unknown-model", prompt_tokens=100, completion_tokens=50)
        assert cost > 0
