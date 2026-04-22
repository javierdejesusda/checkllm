from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.tool_accuracy import ToolAccuracyMetric
from checkllm.models import JudgeResponse


class TestToolAccuracyMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_with_correct_tools(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="All expected tools were called with correct parameters",
            raw_output="",
        )
        metric = ToolAccuracyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output='[{"tool": "search", "params": {"query": "weather NYC"}}, {"tool": "format", "params": {"style": "brief"}}]',
            expected_tools=[
                {"tool": "search", "params": {"query": "weather NYC"}},
                {"tool": "format", "params": {"style": "brief"}},
            ],
            query="What's the weather in NYC?",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "tool_accuracy"

    @pytest.mark.asyncio
    async def test_fails_with_wrong_tools(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1,
            reasoning="Agent used completely wrong tools",
            raw_output="",
        )
        metric = ToolAccuracyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output='[{"tool": "calculator", "params": {"expression": "2+2"}}]',
            expected_tools=[
                {"tool": "search", "params": {"query": "weather NYC"}},
            ],
            query="What's the weather in NYC?",
        )
        assert result.passed is False
        assert result.score == 0.1
        assert result.metric_name == "tool_accuracy"

    @pytest.mark.asyncio
    async def test_expected_tools_in_prompt(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = ToolAccuracyMetric(judge=mock_judge, threshold=0.8)
        expected_tools = [
            {"tool": "web_search", "params": {"q": "python docs"}},
            {"tool": "read_file", "params": {"path": "/tmp/data.txt"}},
        ]
        await metric.evaluate(
            output="agent trace here",
            expected_tools=expected_tools,
            query="find python docs and read the data file",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "web_search" in prompt
        assert "python docs" in prompt
        assert "read_file" in prompt
        assert "/tmp/data.txt" in prompt
        assert "find python docs and read the data file" in prompt
        assert "agent trace here" in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="some tools correct, some missing", raw_output=""
        )
        metric = ToolAccuracyMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="trace",
            expected_tools=[{"tool": "search", "params": {}}],
            query="test query",
        )
        assert result.passed is True
        assert result.score == 0.6
