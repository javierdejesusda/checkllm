from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.mcp_use import MCPUseMetric
from checkllm.models import JudgeResponse


class TestMCPUseMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="Agent selected exactly the right tools for the query",
            raw_output="",
        )
        metric = MCPUseMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Here are the search results for Python documentation.",
            tools_available=["web_search", "file_read", "code_execute", "database_query"],
            tools_used=["web_search"],
            query="Search for Python official documentation",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "mcp_use"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1,
            reasoning="Agent used database_query instead of web_search, which is inappropriate",
            raw_output="",
        )
        metric = MCPUseMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="No relevant records found in the database.",
            tools_available=["web_search", "file_read", "database_query"],
            tools_used=["database_query"],
            query="Find the latest news about AI regulations",
        )
        assert result.passed is False
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Acceptable but not optimal tool selection", raw_output=""
        )
        metric = MCPUseMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            output="Partial answer retrieved.",
            tools_available=["web_search", "file_read"],
            tools_used=["file_read"],
            query="Find recent articles about climate change",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = MCPUseMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            tools_available=["tool_a", "tool_b"],
            tools_used=["tool_a"],
            query="test query",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test output" in prompt
        assert "test query" in prompt
        assert "tool_a" in prompt
