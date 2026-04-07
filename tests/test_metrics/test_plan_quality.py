from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.plan_quality import PlanQualityMetric
from checkllm.models import JudgeResponse


class TestPlanQualityMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Plan is logical and complete", raw_output=""
        )
        metric = PlanQualityMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            plan="1. Gather requirements\n2. Design solution\n3. Implement",
            task="Build a REST API",
        )
        assert result.passed
        assert result.metric_name == "plan_quality"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Plan is missing critical steps", raw_output=""
        )
        metric = PlanQualityMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(plan="Just do it", task="Build a REST API")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Adequate plan", raw_output=""
        )
        metric = PlanQualityMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(plan="Some plan", task="Some task")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = PlanQualityMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "evaluate" in metric.system_prompt.lower() or "assess" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_plan_and_task(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = PlanQualityMetric(judge=mock_judge)
        await metric.evaluate(plan="my plan here", task="my task here")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my plan here" in prompt
        assert "my task here" in prompt
