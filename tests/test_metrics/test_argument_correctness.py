from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.argument_correctness import ArgumentCorrectnessMetric
from checkllm.models import JudgeResponse


class TestArgumentCorrectnessMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Tool calls match expected", raw_output=""
        )
        metric = ArgumentCorrectnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            tool_calls='search(query="python tutorial")',
            expected_calls='search(query="python tutorial")',
        )
        assert result.passed
        assert result.metric_name == "argument_correctness"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.1, reasoning="Wrong tool called", raw_output=""
        )
        metric = ArgumentCorrectnessMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            tool_calls="delete(id=123)",
            expected_calls='search(query="python")',
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Mostly correct", raw_output=""
        )
        metric = ArgumentCorrectnessMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(tool_calls="call_a()", expected_calls="call_a()")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = ArgumentCorrectnessMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "argument" in metric.system_prompt.lower() or "tool" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_calls(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = ArgumentCorrectnessMetric(judge=mock_judge)
        await metric.evaluate(tool_calls="actual_tool(x=1)", expected_calls="expected_tool(x=1)")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "actual_tool(x=1)" in prompt
        assert "expected_tool(x=1)" in prompt
