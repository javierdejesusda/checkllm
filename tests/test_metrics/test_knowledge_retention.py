from unittest.mock import AsyncMock

import pytest

from checkllm.conversation import Turn, ConversationalTestCase
from checkllm.metrics.knowledge_retention import KnowledgeRetentionMetric
from checkllm.models import JudgeResponse


class TestKnowledgeRetentionMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.fixture
    def conversation(self):
        return ConversationalTestCase(
            turns=[
                Turn(role="user", content="My name is Alice and I like cats"),
                Turn(role="assistant", content="Nice to meet you, Alice!"),
                Turn(role="user", content="What's my name?"),
                Turn(role="assistant", content="Your name is Alice!"),
            ]
        )

    @pytest.mark.asyncio
    async def test_passes_when_knowledge_retained(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="The assistant correctly recalled that the user's name is Alice.",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "knowledge_retention"

    @pytest.mark.asyncio
    async def test_fails_when_knowledge_forgotten(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3,
            reasoning="The assistant failed to remember the user's name.",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.passed is False
        assert result.score == 0.3
        assert result.metric_name == "knowledge_retention"

    @pytest.mark.asyncio
    async def test_transcript_in_prompt(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)

        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        # The prompt should contain the formatted transcript
        assert "[USER]: My name is Alice and I like cats" in prompt
        assert "[ASSISTANT]: Nice to meet you, Alice!" in prompt
        assert "[USER]: What's my name?" in prompt
        assert "[ASSISTANT]: Your name is Alice!" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_judge(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)

        call_args = mock_judge.evaluate.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "knowledge retention" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_uses_custom_threshold(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6,
            reasoning="partial retention",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(conversation)
        assert result.passed is True
        assert result.score == 0.6

    @pytest.mark.asyncio
    async def test_cost_from_judge_response(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
            cost=0.005,
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert result.cost == 0.005

    @pytest.mark.asyncio
    async def test_reasoning_from_judge(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.85,
            reasoning="The assistant remembered the user's name and preference for cats.",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(conversation)
        assert "remembered" in result.reasoning
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_called_once(self, mock_judge, conversation):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="ok",
            raw_output="",
        )
        metric = KnowledgeRetentionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(conversation)
        mock_judge.evaluate.assert_awaited_once()
