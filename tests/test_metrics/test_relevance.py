from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.relevance import RelevanceMetric
from checkllm.models import JudgeResponse


class TestRelevanceMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_passes_when_relevant(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Output directly answers the query", raw_output=""
        )
        metric = RelevanceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a programming language.",
            query="What is Python?",
        )
        assert result.passed is True
        assert result.metric_name == "relevance"

    @pytest.mark.asyncio
    async def test_fails_when_irrelevant(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1, reasoning="Output does not address the query", raw_output=""
        )
        metric = RelevanceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="The weather is nice today.",
            query="What is Python?",
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_query(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = RelevanceMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(output="my output", query="my query")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my output" in prompt
        assert "my query" in prompt
