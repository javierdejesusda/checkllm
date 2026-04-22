from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.multi_turn_mcp_use import MultiTurnMCPUseMetric
from checkllm.models import JudgeResponse


class TestMultiTurnMCPUseMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.93,
            reasoning="Agent consistently selected appropriate tools across all turns",
            raw_output="",
        )
        metric = MultiTurnMCPUseMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            conversation_trace=(
                "Turn 1 - User: Search for Python docs\n"
                "  Tool: web_search(query='Python documentation')\n"
                "Turn 2 - User: Now get the file locally\n"
                "  Tool: file_read(path='docs/python.md')\n"
            ),
            tools_available=["web_search", "file_read", "code_execute"],
            tools_used=["web_search", "file_read"],
        )
        assert result.passed is True
        assert result.score == 0.93
        assert result.metric_name == "multi_turn_mcp_use"

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.15,
            reasoning="Agent used wrong tools throughout the conversation",
            raw_output="",
        )
        metric = MultiTurnMCPUseMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            conversation_trace=(
                "Turn 1 - User: Search for information online\n"
                "  Tool: code_execute(code='print(42)')\n"
                "Turn 2 - User: Look up a file\n"
                "  Tool: code_execute(code='print(hello)')\n"
            ),
            tools_available=["web_search", "file_read", "code_execute"],
            tools_used=["code_execute"],
        )
        assert result.passed is False
        assert result.score == 0.15

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Mixed tool selection quality", raw_output=""
        )
        metric = MultiTurnMCPUseMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(
            conversation_trace="Turn 1 - some tool call\nTurn 2 - another tool call",
            tools_available=["web_search", "file_read"],
            tools_used=["web_search"],
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_inputs(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = MultiTurnMCPUseMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            conversation_trace="test conversation trace",
            tools_available=["tool_x", "tool_y"],
            tools_used=["tool_x"],
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "test conversation trace" in prompt
        assert "tool_x" in prompt
