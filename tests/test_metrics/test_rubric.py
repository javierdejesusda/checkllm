from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.rubric import RubricMetric
from checkllm.models import JudgeResponse


class TestRubricMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_passes_when_criteria_met(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Output meets all criteria", raw_output=""
        )
        metric = RubricMetric(judge=mock_judge)
        result = await metric.evaluate(
            output="Python is a high-level language.",
            criteria="concise, under 2 sentences",
            threshold=0.8,
        )
        assert result.passed is True
        assert result.metric_name == "rubric"

    @pytest.mark.asyncio
    async def test_fails_when_criteria_not_met(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Output is too verbose", raw_output=""
        )
        metric = RubricMetric(judge=mock_judge)
        result = await metric.evaluate(
            output="A very long rambling output...",
            criteria="concise, under 2 sentences",
            threshold=0.8,
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_criteria(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = RubricMetric(judge=mock_judge)
        await metric.evaluate(
            output="test", criteria="must mention Python", threshold=0.8
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "must mention Python" in prompt

    @pytest.mark.asyncio
    async def test_uses_per_call_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="partial", raw_output=""
        )
        metric = RubricMetric(judge=mock_judge)
        result_low = await metric.evaluate(
            output="test", criteria="test", threshold=0.5
        )
        result_high = await metric.evaluate(
            output="test", criteria="test", threshold=0.9
        )
        assert result_low.passed is True
        assert result_high.passed is False
