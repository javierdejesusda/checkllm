from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.step_efficiency import StepEfficiencyMetric
from checkllm.models import JudgeResponse


class TestStepEfficiencyMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Steps were efficient", raw_output=""
        )
        metric = StepEfficiencyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            steps=["Read the file", "Parse the JSON", "Write output"],
            task="Parse and transform a JSON file",
        )
        assert result.passed
        assert result.metric_name == "step_efficiency"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Many redundant steps", raw_output=""
        )
        metric = StepEfficiencyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            steps=["Do X", "Redo X", "Redo X again", "Do Y"],
            task="Do X then Y",
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Some redundancy", raw_output=""
        )
        metric = StepEfficiencyMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(steps=["step1", "step2"], task="task")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = StepEfficiencyMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert (
            "redundant" in metric.system_prompt.lower()
            or "efficient" in metric.system_prompt.lower()
        )

    @pytest.mark.asyncio
    async def test_steps_numbered_in_prompt(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = StepEfficiencyMetric(judge=mock_judge)
        await metric.evaluate(steps=["alpha step", "beta step"], task="some task")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "1. alpha step" in prompt
        assert "2. beta step" in prompt
        assert "some task" in prompt
