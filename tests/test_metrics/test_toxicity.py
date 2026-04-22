from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.toxicity import ToxicityMetric
from checkllm.models import JudgeResponse


class TestToxicityMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_passes_when_safe(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.98, reasoning="Output is safe and appropriate", raw_output=""
        )
        metric = ToxicityMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="Have a great day!")
        assert result.passed is True
        assert result.metric_name == "toxicity"

    @pytest.mark.asyncio
    async def test_fails_when_toxic(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1, reasoning="Output contains harmful content", raw_output=""
        )
        metric = ToxicityMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="toxic content here")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_output(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = ToxicityMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(output="some output text")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "some output text" in prompt
