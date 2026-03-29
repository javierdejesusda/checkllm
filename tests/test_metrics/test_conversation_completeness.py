from unittest.mock import AsyncMock

import pytest

from checkllm.conversation import Turn, ConversationalTestCase
from checkllm.metrics.conversation_completeness import ConversationCompletenessMetric
from checkllm.models import JudgeResponse


class TestConversationCompletenessMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.fixture
    def conversation(self):
        return ConversationalTestCase(
            turns=[
                Turn(role="user", content="Tell me about Python and Java"),
                Turn(role="assistant", content="Python is a high-level language. Java is a statically typed language."),
                Turn(role="user", content="Which one is faster?"),
                Turn(role="assistant", content="Java is generally faster due to JIT compilation."),
            ]
        )

    @pytest.mark.asyncio
    async def test_passes_when_all_addressed(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="All user requests were fully addressed.",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "conversation_completeness"

    @pytest.mark.asyncio
    async def test_fails_when_requests_missed(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.4,
            reasoning="The assistant only addressed Python but ignored Java.",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.passed is False
        assert result.score == 0.4
        assert result.metric_name == "conversation_completeness"

    @pytest.mark.asyncio
    async def test_transcript_in_prompt(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)

        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        # The prompt should contain the formatted transcript
        assert "[USER]: Tell me about Python and Java" in prompt
        assert "[ASSISTANT]: Python is a high-level language." in prompt
        assert "[USER]: Which one is faster?" in prompt
        assert "[ASSISTANT]: Java is generally faster" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_judge(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)

        call_args = mock_judge.evaluate.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "completeness" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_uses_custom_threshold(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6,
            reasoning="partial completeness",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(conversation)
        assert result.passed is True
        assert result.score == 0.6

    @pytest.mark.asyncio
    async def test_cost_from_judge_response(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
            cost=0.003,
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.cost == 0.003

    @pytest.mark.asyncio
    async def test_reasoning_from_judge(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.85,
            reasoning="Both Python and Java topics were addressed and speed comparison provided.",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert "Python" in result.reasoning
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_called_once(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)
        mock_judge.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prompt_mentions_completeness_task(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = ConversationCompletenessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)

        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        # The prompt should instruct the judge to evaluate completeness
        assert "request" in prompt.lower() or "question" in prompt.lower()
        assert "addressed" in prompt.lower() or "completeness" in prompt.lower()
