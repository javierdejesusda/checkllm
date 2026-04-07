from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.g_eval import GEvalMetric
from checkllm.models import JudgeResponse


class TestGEvalMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_with_high_score(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Output meets all criteria", raw_output=""
        )
        metric = GEvalMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Paris is the capital of France.",
            criteria="Factual accuracy",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "g_eval"

    @pytest.mark.asyncio
    async def test_fails_with_low_score(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Output fails to meet criteria", raw_output=""
        )
        metric = GEvalMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="I don't know.",
            criteria="Factual accuracy and completeness",
        )
        assert result.passed is False
        assert result.score == 0.3
        assert result.metric_name == "g_eval"

    @pytest.mark.asyncio
    async def test_custom_steps_included_in_prompt(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.85, reasoning="ok", raw_output=""
        )
        steps = ["Check grammar", "Check tone", "Check accuracy"]
        metric = GEvalMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            criteria="writing quality",
            steps=steps,
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Check grammar" in prompt
        assert "Check tone" in prompt
        assert "Check accuracy" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt
        assert "Step 3" in prompt
        assert "writing quality" in prompt
        assert "test output" in prompt

    @pytest.mark.asyncio
    async def test_no_steps_generates_own(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.7, reasoning="decent output", raw_output=""
        )
        metric = GEvalMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="some output",
            criteria="coherence",
            steps=None,
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "No evaluation steps were provided" in prompt
        assert "Generate your own" in prompt
        assert "coherence" in prompt
        assert "some output" in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="partially meets criteria", raw_output=""
        )
        metric = GEvalMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="test", criteria="test criteria"
        )
        assert result.passed is True
        assert result.score == 0.6
