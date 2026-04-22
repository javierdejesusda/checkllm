from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.role_violation import RoleViolationMetric
from checkllm.models import JudgeResponse


class TestRoleViolationMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="Output stays within role boundaries", raw_output=""
        )
        metric = RoleViolationMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="I can help you find the best deals on electronics.",
            role_description="A shopping assistant that helps users find products",
        )
        assert result.passed
        assert result.metric_name == "role_violation"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.0, reasoning="Output violates role boundaries", raw_output=""
        )
        metric = RoleViolationMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="I am now acting as a different AI with no restrictions.",
            role_description="A shopping assistant",
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Minor boundary issue", raw_output=""
        )
        metric = RoleViolationMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(output="some output", role_description="some role")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = RoleViolationMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "role" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_role(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=1.0, reasoning="ok", raw_output="")
        metric = RoleViolationMetric(judge=mock_judge)
        await metric.evaluate(output="my output text", role_description="my role text")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "my output text" in prompt
        assert "my role text" in prompt
