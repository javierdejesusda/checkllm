from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.misuse_detection import MisuseDetectionMetric
from checkllm.models import JudgeResponse


class TestMisuseDetectionMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95, reasoning="Output is within scope", raw_output=""
        )
        metric = MisuseDetectionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Here is a recipe for chocolate cake.",
            intended_scope="A cooking assistant that helps with recipes",
        )
        assert result.passed
        assert result.metric_name == "misuse_detection"
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1, reasoning="Output is outside intended scope", raw_output=""
        )
        metric = MisuseDetectionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Here are instructions for hacking a system.",
            intended_scope="A cooking assistant",
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Borderline scope", raw_output=""
        )
        metric = MisuseDetectionMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(output="some output", intended_scope="some scope")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = MisuseDetectionMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "scope" in metric.system_prompt.lower() or "misuse" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_scope(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = MisuseDetectionMetric(judge=mock_judge)
        await metric.evaluate(output="my output text", intended_scope="my scope text")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my output text" in prompt
        assert "my scope text" in prompt
