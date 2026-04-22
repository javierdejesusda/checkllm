from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.image_helpfulness import ImageHelpfulnessMetric
from checkllm.models import JudgeResponse


class TestImageHelpfulnessMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.92,
            reasoning="Image directly answers the user's question",
            raw_output="",
        )
        metric = ImageHelpfulnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A step-by-step diagram illustrating how to tie a bowline knot",
            query="How do I tie a bowline knot?",
        )
        assert result.passed is True
        assert result.score == 0.92
        assert result.metric_name == "image_helpfulness"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Image does not help answer the query", raw_output=""
        )
        metric = ImageHelpfulnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A logo of a company",
            query="What are the steps to deploy a Kubernetes cluster?",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Somewhat helpful", raw_output=""
        )
        metric = ImageHelpfulnessMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            image_description="A related but indirect illustration",
            query="Explain recursion",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = ImageHelpfulnessMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            image_description="test image description",
            query="test user query",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test image description" in prompt
        assert "test user query" in prompt
