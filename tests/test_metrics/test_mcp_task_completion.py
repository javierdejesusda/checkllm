from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.mcp_task_completion import MCPTaskCompletionMetric
from checkllm.models import JudgeResponse


class TestMCPTaskCompletionMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9,
            reasoning="Task fully completed using appropriate tools",
            raw_output="",
        )
        metric = MCPTaskCompletionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Successfully fetched and summarized 10 recent GitHub issues.",
            task="Fetch recent GitHub issues and provide a summary",
            tools_used=["github_list_issues", "github_get_issue"],
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "mcp_task_completion"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2,
            reasoning="Agent failed to complete the task; output does not address requirements",
            raw_output="",
        )
        metric = MCPTaskCompletionMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="I am unable to complete this task.",
            task="Create a new file with specified content in the repository",
            tools_used=[],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Partially completed", raw_output=""
        )
        metric = MCPTaskCompletionMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            output="Retrieved some files but did not process them all.",
            task="Process all files in the directory",
            tools_used=["filesystem_list", "filesystem_read"],
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = MCPTaskCompletionMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test agent output",
            task="test task description",
            tools_used=["test_tool"],
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test agent output" in prompt
        assert "test task description" in prompt
        assert "test_tool" in prompt
