"""Tests for the fluency metric."""

from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.fluency import FluencyMetric
from checkllm.models import JudgeResponse


class TestFluencyMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Very fluent output", cost=0.001
        )
        return judge

    @pytest.mark.asyncio
    async def test_evaluate_passing(self, mock_judge):
        metric = FluencyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="This is a fluent sentence.")
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "fluency"

    @pytest.mark.asyncio
    async def test_evaluate_failing(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Very awkward phrasing", cost=0.001
        )
        metric = FluencyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="Bad grammar this is.")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_judge):
        metric = FluencyMetric(judge=mock_judge)
        metric.system_prompt = "Custom prompt"
        await metric.evaluate(output="test")
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom prompt"
