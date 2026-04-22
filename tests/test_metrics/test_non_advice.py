from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.non_advice import NonAdviceMetric
from checkllm.models import JudgeResponse


class TestNonAdviceMetric:
    @pytest.fixture
    def mock_judge(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_pass(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="No restricted advice given", raw_output=""
        )
        metric = NonAdviceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(output="I recommend consulting a doctor for your symptoms.")
        assert result.passed
        assert result.metric_name == "non_advice"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_fail(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.0, reasoning="Gives specific medical diagnosis", raw_output=""
        )
        metric = NonAdviceMetric(judge=mock_judge, threshold=0.8)
        result = await metric.evaluate(
            output="Based on your symptoms, you have type 2 diabetes. Take metformin 500mg twice daily."
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.5, reasoning="Borderline", raw_output=""
        )
        metric = NonAdviceMetric(judge=mock_judge, threshold=0.4)
        result = await metric.evaluate(output="some output")
        assert result.passed

    def test_prompt_content(self, mock_judge):
        metric = NonAdviceMetric(judge=mock_judge)
        assert "score" in metric.system_prompt.lower()
        assert "advice" in metric.system_prompt.lower() or "domain" in metric.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_default_domains_in_prompt(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=1.0, reasoning="ok", raw_output="")
        metric = NonAdviceMetric(judge=mock_judge)
        await metric.evaluate(output="some output text")
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "medical" in prompt
        assert "legal" in prompt
        assert "financial" in prompt

    @pytest.mark.asyncio
    async def test_custom_domains(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=1.0, reasoning="ok", raw_output="")
        metric = NonAdviceMetric(judge=mock_judge)
        await metric.evaluate(output="some output", restricted_domains=["nutrition", "fitness"])
        call_args = mock_judge.evaluate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "nutrition" in prompt
        assert "fitness" in prompt
