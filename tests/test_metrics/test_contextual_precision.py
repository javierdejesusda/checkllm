from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.contextual_precision import ContextualPrecisionMetric
from checkllm.models import CheckResult, JudgeResponse


class TestContextualPrecisionMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_when_relevant_docs_ranked_high(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="Relevant documents are ranked at the top",
            raw_output="",
        )
        metric = ContextualPrecisionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Paris is the capital of France.",
            context=[
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
                "The Eiffel Tower is in Paris.",
            ],
            query="What is the capital of France?",
            expected="Paris",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "contextual_precision"

    @pytest.mark.asyncio
    async def test_fails_when_relevant_docs_ranked_low(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2,
            reasoning="Relevant documents are buried at the bottom",
            raw_output="",
        )
        metric = ContextualPrecisionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Paris is the capital of France.",
            context=[
                "Pizza is popular in Italy.",
                "The weather is nice today.",
                "Paris is the capital of France.",
            ],
            query="What is the capital of France?",
            expected="Paris",
        )
        assert result.passed is False
        assert result.score == 0.2
        assert result.metric_name == "contextual_precision"

    @pytest.mark.asyncio
    async def test_prompt_contains_documents_and_query(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = ContextualPrecisionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            context=["first document", "second document"],
            query="test query",
            expected="test expected",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test query" in prompt
        assert "test expected" in prompt
        assert "first document" in prompt
        assert "second document" in prompt
        assert "Document 1" in prompt
        assert "Document 2" in prompt
        assert "test output" in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="moderate ranking quality", raw_output=""
        )
        metric = ContextualPrecisionMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="test",
            context=["doc1"],
            query="query",
            expected="expected",
        )
        assert result.passed is True
        assert result.score == 0.6
