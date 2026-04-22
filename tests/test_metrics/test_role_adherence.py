from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.role_adherence import RoleAdherenceMetric
from checkllm.models import JudgeResponse


class TestRoleAdherenceMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_passes_when_in_role(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.95,
            reasoning="Output maintains the pirate persona consistently throughout",
            raw_output="",
        )
        metric = RoleAdherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Arrr matey! Ye be askin' about the treasure map, aye?",
            role_description="You are a friendly pirate captain. Always speak in pirate dialect.",
            query="Where is the treasure?",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "role_adherence"

    @pytest.mark.asyncio
    async def test_fails_when_out_of_role(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.15,
            reasoning="Output completely drops the pirate persona and responds as a generic AI",
            raw_output="",
        )
        metric = RoleAdherenceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="As an AI language model, I can help you find information about treasures.",
            role_description="You are a friendly pirate captain. Always speak in pirate dialect.",
            query="Where is the treasure?",
        )
        assert result.passed is False
        assert result.score == 0.15
        assert result.metric_name == "role_adherence"

    @pytest.mark.asyncio
    async def test_query_included_when_provided(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        metric = RoleAdherenceMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            role_description="a helpful teacher",
            query="explain quantum physics",
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "a helpful teacher" in prompt
        assert "explain quantum physics" in prompt
        assert "test output" in prompt
        assert "User Query" in prompt

    @pytest.mark.asyncio
    async def test_no_query(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.85, reasoning="role maintained", raw_output=""
        )
        metric = RoleAdherenceMetric(judge=mock_judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            role_description="a stern military officer",
            query=None,
        )
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "a stern military officer" in prompt
        assert "test output" in prompt
        # When no query is provided, "User Query" should not appear
        assert "User Query" not in prompt

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="moderate role adherence", raw_output=""
        )
        metric = RoleAdherenceMetric(judge=mock_judge, threshold=0.5)
        result = await metric.evaluate(
            output="test",
            role_description="a doctor",
        )
        assert result.passed is True
        assert result.score == 0.6
