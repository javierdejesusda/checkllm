from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.image_coherence import ImageCoherenceMetric
from checkllm.models import JudgeResponse


class TestImageCoherenceMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Image aligns well with text", raw_output=""
        )
        metric = ImageCoherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A diagram showing network topology with connected nodes",
            text_context="This section explains how nodes communicate in a distributed network.",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "image_coherence"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Image has no relation to text", raw_output=""
        )
        metric = ImageCoherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            image_description="A photo of a sunset over the ocean",
            text_context="This section explains database indexing strategies.",
        )
        assert result.passed is False
        assert result.score == 0.3

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Somewhat coherent", raw_output=""
        )
        metric = ImageCoherenceMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            image_description="An abstract illustration",
            text_context="A vague description of concepts.",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = ImageCoherenceMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            image_description="test image description",
            text_context="test text context",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test image description" in prompt
        assert "test text context" in prompt
