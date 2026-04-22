from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.pii_detection import PIIDetectionMetric
from checkllm.models import JudgeResponse


class TestPIIDetectionMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="No PII detected", raw_output=""
        )
        metric = PIIDetectionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="The weather today is sunny.")
        assert result.passed
        assert result.metric_name == "pii_detection"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.0, reasoning="Contains name and SSN", raw_output=""
        )
        metric = PIIDetectionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="John Smith, SSN: 123-45-6789, lives at 123 Main St")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Ambiguous reference", raw_output=""
        )
        metric = PIIDetectionMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(output="Some output text")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = PIIDetectionMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "pii" in metric.system_prompt.lower() or "personal" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_output(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=1.0, reasoning="ok", raw_output="")
        metric = PIIDetectionMetric(judge=mock_judge)
        await metric.evaluate(output="unique output content here")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "unique output content here" in prompt
