"""Tests for the native DeepSeekJudge backend.

Mocks the OpenAI async client (pointed at the DeepSeek endpoint) so the
tests run fully offline.  Covers configuration, evaluation, pricing, the
``reasoning_content`` field for ``deepseek-reasoner``, streaming, and the
factory alias ``judge="deepseek-chat"``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.judge import (
    DeepSeekJudge,
    JudgeBackend,
    JudgeConfigError,
    StreamingJudgeResult,
    _DEEPSEEK_PRICES,
    estimate_cost,
)
from checkllm.models import JudgeResponse


def _make_chat_response(
    content: str,
    reasoning_content: str | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a mock ``chat.completions.create`` response."""
    message = MagicMock()
    message.content = content
    message.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


async def _async_chunks(chunks: list[Any]):
    for chunk in chunks:
        yield chunk


def _make_stream_chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    usage: dict[str, int] | None = None,
) -> MagicMock:
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    if usage is not None:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
    else:
        chunk.usage = None
    return chunk


class TestDeepSeekConfig:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(JudgeConfigError, match="DEEPSEEK_API_KEY"):
            DeepSeekJudge()

    def test_constructor_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        judge = DeepSeekJudge()
        assert judge.model == "deepseek-chat"
        assert judge._api_key == "sk-env"
        assert judge._base_url == "https://api.deepseek.com/v1"
        assert isinstance(judge, JudgeBackend)

    def test_constructor_with_explicit_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        judge = DeepSeekJudge(api_key="sk-explicit", model="deepseek-reasoner")
        assert judge._api_key == "sk-explicit"
        assert judge.model == "deepseek-reasoner"

    def test_custom_base_url(self) -> None:
        judge = DeepSeekJudge(api_key="k", base_url="https://proxy.example.com/v2/")
        assert judge._base_url == "https://proxy.example.com/v2"

    def test_client_configured_with_deepseek_endpoint(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        # The underlying AsyncOpenAI client should be pointed at DeepSeek.
        assert str(judge._client.base_url).startswith("https://api.deepseek.com")


class TestDeepSeekEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_parses_score_and_reasoning(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        resp = _make_chat_response(
            content=json.dumps({"score": 0.72, "reasoning": "Mostly correct"}),
        )
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = resp
            result = await judge.evaluate("Evaluate this")

        assert isinstance(result, JudgeResponse)
        assert result.score == pytest.approx(0.72)
        assert result.reasoning == "Mostly correct"

    @pytest.mark.asyncio
    async def test_evaluate_cost_uses_deepseek_pricing(self) -> None:
        judge = DeepSeekJudge(api_key="k", model="deepseek-chat")
        resp = _make_chat_response(
            content=json.dumps({"score": 1.0, "reasoning": "ok"}),
            prompt_tokens=1000,
            completion_tokens=500,
        )
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = resp
            await judge.evaluate("x")

        input_price, output_price = _DEEPSEEK_PRICES["deepseek-chat"]
        expected = 1000 * input_price + 500 * output_price
        assert judge.last_cost == pytest.approx(expected)
        assert judge.total_cost == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_reasoning_content_captured(self) -> None:
        judge = DeepSeekJudge(api_key="k", model="deepseek-reasoner")
        resp = _make_chat_response(
            content=json.dumps({"score": 0.9, "reasoning": "deep"}),
            reasoning_content="step 1\nstep 2",
        )
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = resp
            result = await judge.evaluate("hard")

        assert judge.last_reasoning_content == "step 1\nstep 2"
        assert "<reasoning>" in (result.raw_output or "")
        assert "step 1" in (result.raw_output or "")

    @pytest.mark.asyncio
    async def test_system_prompt_forwarded(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        resp = _make_chat_response(content=json.dumps({"score": 0.5, "reasoning": ""}))
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = resp
            await judge.evaluate("p", system_prompt="You are strict.")
            messages = mock_create.call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are strict."

    @pytest.mark.asyncio
    async def test_malformed_json_returns_zero(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        resp = _make_chat_response(content="not JSON at all")
        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = resp
            result = await judge.evaluate("p")

        assert result.score == 0.0
        assert "parse" in result.reasoning.lower() or "failed" in result.reasoning.lower()


class TestDeepSeekStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks_and_final_result(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        chunks = [
            _make_stream_chunk(content='{"score"'),
            _make_stream_chunk(content=": 0.9, "),
            _make_stream_chunk(content='"reasoning": "good"}'),
            _make_stream_chunk(usage={"prompt_tokens": 20, "completion_tokens": 10}),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _async_chunks(chunks)

            pieces: list[str] = []
            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("hi"):
                if isinstance(item, str):
                    pieces.append(item)
                else:
                    final = item

        assert "".join(pieces) == '{"score": 0.9, "reasoning": "good"}'
        assert final is not None
        assert final.response.score == pytest.approx(0.9)
        assert final.response.reasoning == "good"
        assert final.stopped_early is False

    @pytest.mark.asyncio
    async def test_stream_reasoning_content_accumulated(self) -> None:
        judge = DeepSeekJudge(api_key="k", model="deepseek-reasoner")
        chunks = [
            _make_stream_chunk(reasoning_content="thinking..."),
            _make_stream_chunk(content='{"score": 0.3, "reasoning": "meh"}'),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _async_chunks(chunks)

            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("x"):
                if isinstance(item, StreamingJudgeResult):
                    final = item

        assert judge.last_reasoning_content == "thinking..."
        assert final is not None
        assert "<reasoning>" in (final.response.raw_output or "")

    @pytest.mark.asyncio
    async def test_stream_early_stop_callback(self) -> None:
        judge = DeepSeekJudge(api_key="k")
        chunks = [
            _make_stream_chunk(content="AAA"),
            _make_stream_chunk(content="BBB"),
            _make_stream_chunk(content="STOP"),
            _make_stream_chunk(content="should-not-arrive"),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _async_chunks(chunks)

            seen: list[str] = []
            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("hi", on_token=lambda t: "STOP" in t):
                if isinstance(item, str):
                    seen.append(item)
                else:
                    final = item

        assert "AAA" in seen and "BBB" in seen and "STOP" in seen
        assert "should-not-arrive" not in seen
        assert final is not None
        assert final.stopped_early is True


class TestDeepSeekFactoryIntegration:
    def test_factory_alias_deepseek_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        from checkllm.providers import create_judge

        judge = create_judge("deepseek-chat")
        assert isinstance(judge, DeepSeekJudge)
        assert judge.model == "deepseek-chat"

    def test_factory_alias_deepseek_reasoner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        from checkllm.providers import create_judge

        judge = create_judge("deepseek-reasoner")
        assert isinstance(judge, DeepSeekJudge)
        assert judge.model == "deepseek-reasoner"

    def test_estimate_cost_knows_deepseek_models(self) -> None:
        # Reasoner is pricier than chat for both directions.
        chat_cost = estimate_cost("deepseek-chat", 1_000_000, 0)
        reasoner_cost = estimate_cost("deepseek-reasoner", 1_000_000, 0)
        assert reasoner_cost > chat_cost
