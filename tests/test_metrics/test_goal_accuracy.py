from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.goal_accuracy import GoalAccuracyMetric
from checkllm.models import JudgeResponse


class TestGoalAccuracyMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Output fully achieves the goal", raw_output=""
        )
        metric = GoalAccuracyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="The API is deployed and returning 200 OK",
            goal="Deploy the REST API successfully",
        )
        assert result.passed
        assert result.metric_name == "goal_accuracy"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Output does not achieve the goal", raw_output=""
        )
        metric = GoalAccuracyMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="Nothing happened", goal="Deploy the API")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Partially achieved", raw_output=""
        )
        metric = GoalAccuracyMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(output="Some output", goal="Some goal")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = GoalAccuracyMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_goal(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = GoalAccuracyMetric(judge=mock_judge)
        await metric.evaluate(output="my output here", goal="my goal here")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my output here" in prompt
        assert "my goal here" in prompt
