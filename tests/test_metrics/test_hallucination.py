from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.hallucination import HallucinationMetric
from checkllm.models import JudgeResponse


class TestHallucinationMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_when_grounded(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95, reasoning="All claims are supported by context", raw_output=""
        )
        metric = HallucinationMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="The sky is blue.",
            context="The sky appears blue due to Rayleigh scattering.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "hallucination"

    @pytest.mark.asyncio
    async def test_fails_when_hallucinated(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Output contains unsupported claims", raw_output=""
        )
        metric = HallucinationMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="The sky is green and made of cheese.",
            context="The sky appears blue.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_context(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = HallucinationMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(output="test output", context="test context")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test output" in prompt
        assert "test context" in prompt

    @pytest.mark.asyncio
    async def test_uses_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="partially grounded", raw_output=""
        )
        metric = HallucinationMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(output="test", context="test")
        assert result.passed is True
