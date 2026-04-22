from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.text_to_image import TextToImageMetric
from checkllm.models import JudgeResponse


class TestTextToImageMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.88,
            reasoning="Generated image closely matches all elements of the prompt",
            raw_output="",
        )
        metric = TextToImageMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A photorealistic red sports car parked on a mountain road at sunset",
            original_prompt="A red sports car on a mountain road during golden hour, photorealistic",
        )
        assert result.passed is True
        assert result.score == 0.88
        assert result.metric_name == "text_to_image"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.25,
            reasoning="Image shows a blue sedan in a city, not matching the prompt",
            raw_output="",
        )
        metric = TextToImageMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A blue sedan parked in a city street at night",
            original_prompt="A red sports car on a mountain road during golden hour, photorealistic",
        )
        assert result.passed is False
        assert result.score == 0.25

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Partially matches the prompt", raw_output=""
        )
        metric = TextToImageMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            image_description="A car on a road",
            original_prompt="A red sports car on a mountain road during golden hour",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = TextToImageMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            image_description="test generated image",
            original_prompt="test original prompt",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test generated image" in prompt
        assert "test original prompt" in prompt
