"""Tests for the correctness metric."""
from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.correctness import CorrectnessMetric
from checkllm.models import JudgeResponse


class TestCorrectnessMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Semantically equivalent to expected answer", cost=0.001
        )
        return judge

    @pytest.mark.asyncio
    async def test_evaluate_correct(self, mock_judge):
        metric = CorrectnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a high-level programming language",
            expected="Python is a programming language",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "correctness"

    @pytest.mark.asyncio
    async def test_evaluate_incorrect(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Incorrect answer", cost=0.001
        )
        metric = CorrectnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a snake",
            expected="Python is a programming language",
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_judge):
        metric = CorrectnessMetric(judge=mock_judge)
        metric.system_prompt = "Custom correctness prompt"
        await metric.evaluate(output="test", expected="test")
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom correctness prompt"

    @pytest.mark.asyncio
    async def test_prompt_includes_expected(self, mock_judge):
        metric = CorrectnessMetric(judge=mock_judge)
        await metric.evaluate(output="my output", expected="expected answer")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
        assert "expected answer" in prompt
        assert "my output" in prompt
