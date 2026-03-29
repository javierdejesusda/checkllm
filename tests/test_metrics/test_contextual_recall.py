from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.contextual_recall import ContextualRecallMetric
from checkllm.models import CheckResult, JudgeResponse


class TestContextualRecallMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_when_answer_supported(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="All claims in the expected answer are supported by context",
            raw_output="",
        )
        metric = ContextualRecallMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Photosynthesis converts sunlight into chemical energy.",
            context=[
                "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
                "This process occurs in the chloroplasts of plant cells.",
            ],
            expected="Photosynthesis converts sunlight into chemical energy in chloroplasts.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "contextual_recall"

    @pytest.mark.asyncio
    async def test_fails_when_answer_not_supported(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2,
            reasoning="Most claims in the expected answer are not found in context",
            raw_output="",
        )
        metric = ContextualRecallMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="The mitochondria is the powerhouse of the cell.",
            context=[
                "The weather in London is often rainy.",
                "Big Ben is a famous landmark.",
            ],
            expected="Mitochondria produce ATP through cellular respiration.",
        )
        assert result.passed is False
        assert result.score == 0.2
        assert result.metric_name == "contextual_recall"

    @pytest.mark.asyncio
    async def test_prompt_contains_expected_and_context(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = ContextualRecallMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            context=["context doc one", "context doc two"],
            expected="the expected answer",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "the expected answer" in prompt
        assert "context doc one" in prompt
        assert "context doc two" in prompt
        assert "Document 1" in prompt
        assert "Document 2" in prompt
        assert "test output" in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="partial recall", raw_output=""
        )
        metric = ContextualRecallMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="test",
            context=["doc1"],
            expected="expected",
        )
        assert result.passed is True
        assert result.score == 0.6
