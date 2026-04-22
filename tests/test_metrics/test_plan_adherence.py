from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.plan_adherence import PlanAdherenceMetric
from checkllm.models import JudgeResponse


class TestPlanAdherenceMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95, reasoning="Execution followed the plan exactly", raw_output=""
        )
        metric = PlanAdherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            plan="Step 1: fetch data\nStep 2: process\nStep 3: save",
            execution_trace="Fetched data, processed, saved to disk",
        )
        assert result.passed
        assert result.metric_name == "plan_adherence"
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.2, reasoning="Step 2 was skipped entirely", raw_output=""
        )
        metric = PlanAdherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            plan="Step 1\nStep 2\nStep 3",
            execution_trace="Did step 1 then step 3",
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Minor deviations", raw_output=""
        )
        metric = PlanAdherenceMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(plan="plan text", execution_trace="trace text")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = PlanAdherenceMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "plan" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_plan_and_trace(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = PlanAdherenceMetric(judge=mock_judge)
        await metric.evaluate(plan="my plan content", execution_trace="my execution trace")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my plan content" in prompt
        assert "my execution trace" in prompt
