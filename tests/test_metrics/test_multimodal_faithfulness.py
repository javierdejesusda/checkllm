from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.multimodal_faithfulness import MultimodalFaithfulnessMetric
from checkllm.models import JudgeResponse


class TestMultimodalFaithfulnessMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.91,
            reasoning="Text accurately describes all elements shown in the image",
            raw_output="",
        )
        metric = MultimodalFaithfulnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A pie chart with three segments: 50% blue, 30% red, 20% green",
            text_output="The chart shows that blue accounts for half, red for nearly a third, and green for the remainder.",
            source_context="Annual budget allocation overview",
        )
        assert result.passed is True
        assert result.score == 0.91
        assert result.metric_name == "multimodal_faithfulness"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2,
            reasoning="Text describes a bar chart but the image shows a line graph",
            raw_output="",
        )
        metric = MultimodalFaithfulnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A line graph showing upward trend over 12 months",
            text_output="The bar chart shows revenue declining quarter over quarter.",
            source_context="Financial report Q4",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Partially faithful", raw_output=""
        )
        metric = MultimodalFaithfulnessMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            image_description="A simple diagram",
            text_output="The diagram depicts a process flow.",
            source_context="Process documentation",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = MultimodalFaithfulnessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            image_description="test image content",
            text_output="test text output",
            source_context="test source context",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test image content" in prompt
        assert "test text output" in prompt
        assert "test source context" in prompt
