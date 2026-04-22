"""Tests for the coherence metric."""

from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.coherence import CoherenceMetric
from checkllm.models import JudgeResponse


class TestCoherenceMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=0.85, reasoning="Well-structured and logical", cost=0.001
        )
        return judge

    @pytest.mark.asyncio
    async def test_evaluate_passing(self, mock_judge):
        metric = CoherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="First point. Second point follows logically.")
        assert result.passed is True
        assert result.score == 0.85
        assert result.metric_name == "coherence"

    @pytest.mark.asyncio
    async def test_evaluate_failing(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Contradictory and disorganized", cost=0.001
        )
        metric = CoherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="Dogs are cats. The sky is underground.")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_judge):
        metric = CoherenceMetric(judge=mock_judge)
        metric.system_prompt = "Custom coherence prompt"
        await metric.evaluate(output="test")
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom coherence prompt"
