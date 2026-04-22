from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.image_relevance import ImageRelevanceMetric
from checkllm.models import JudgeResponse


class TestImageRelevanceMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="Image is directly relevant to the topic",
            raw_output="",
        )
        metric = ImageRelevanceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A bar chart comparing CPU usage across different programming languages",
            query="Performance benchmarks for programming languages",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "image_relevance"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.15, reasoning="Image has no relation to the query", raw_output=""
        )
        metric = ImageRelevanceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A photo of a cat sitting on a keyboard",
            query="Machine learning model training techniques",
        )
        assert result.passed is False
        assert result.score == 0.15

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Loosely relevant", raw_output=""
        )
        metric = ImageRelevanceMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            image_description="A generic technology image",
            query="Software development practices",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = ImageRelevanceMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            image_description="test image description",
            query="test query topic",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test image description" in prompt
        assert "test query topic" in prompt
