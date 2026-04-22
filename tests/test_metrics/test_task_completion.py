from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.task_completion import TaskCompletionMetric
from checkllm.models import JudgeResponse


class TestTaskCompletionMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_when_task_completed(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="Task fully completed with all constraints satisfied",
            raw_output="",
        )
        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="def add(a, b):\n    return a + b",
            task_description="Write a Python function that adds two numbers",
            constraints=["Must be a pure function", "Must handle integers"],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "task_completion"

    @pytest.mark.asyncio
    async def test_fails_when_task_not_completed(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2,
            reasoning="Output does not accomplish the stated goal",
            raw_output="",
        )
        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="I'm not sure how to do that.",
            task_description="Write a sorting algorithm in Python",
        )
        assert result.passed is False
        assert result.score == 0.2
        assert result.metric_name == "task_completion"

    @pytest.mark.asyncio
    async def test_constraints_included_in_prompt(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            task_description="build a web server",
            constraints=["must use HTTPS", "must handle 1000 concurrent users"],
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "build a web server" in prompt
        assert "must use HTTPS" in prompt
        assert "must handle 1000 concurrent users" in prompt
        assert "test output" in prompt

    @pytest.mark.asyncio
    async def test_no_constraints(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.85, reasoning="task done", raw_output=""
        )
        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            task_description="summarize the article",
            constraints=None,
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "summarize the article" in prompt
        assert "test output" in prompt
        # When no constraints, the word "Constraints" should not appear as a section
        assert "Constraints:" not in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="partially completed", raw_output=""
        )
        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="test",
            task_description="do something",
        )
        assert result.passed is True
        assert result.score == 0.6
