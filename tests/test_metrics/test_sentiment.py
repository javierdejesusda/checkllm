"""Tests for the sentiment metric."""
from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.sentiment import SentimentMetric
from checkllm.models import JudgeResponse


class TestSentimentMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="Positive and encouraging tone", cost=0.001
        )
        return judge

    @pytest.mark.asyncio
    async def test_evaluate_positive(self, mock_judge):
        metric = SentimentMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(output="This is wonderful and exciting!")
        assert result.passed is True
        assert result.score == 0.8
        assert result.metric_name == "sentiment"

    @pytest.mark.asyncio
    async def test_evaluate_negative(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Negative and critical tone", cost=0.001
        )
        metric = SentimentMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(output="This is terrible and awful.")
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_judge):
        metric = SentimentMetric(judge=mock_judge)
        metric.system_prompt = "Custom sentiment prompt"
        await metric.evaluate(output="test")
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom sentiment prompt"
