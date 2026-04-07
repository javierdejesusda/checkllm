from __future__ import annotations

import pytest

from checkllm.metrics.instruction_completeness import InstructionCompletenessMetric
from checkllm.testing import MockJudge


class TestInstructionCompletenessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_all_instructions_followed(self, judge):
        judge.set_default(score=0.95, reasoning="All instructions fully followed")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output=(
                "Dear Customer,\n\n"
                "Thank you for reaching out. I understand your concern about the billing issue. "
                "Your refund of $50 has been processed and will appear within 3-5 business days.\n\n"
                "Best regards,\nSupport Team"
            ),
            instructions=[
                "Address the customer formally",
                "Acknowledge their concern",
                "Provide the refund amount",
                "Include a timeline for the refund",
            ],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "instruction_completeness"

    @pytest.mark.asyncio
    async def test_fails_when_instructions_missed(self, judge):
        judge.set_default(score=0.25, reasoning="Only 1 of 4 instructions followed")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Your refund has been processed.",
            instructions=[
                "Address the customer formally",
                "Acknowledge their concern",
                "Provide the refund amount",
                "Include a timeline for the refund",
            ],
        )
        assert result.passed is False
        assert result.score == 0.25

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", instructions=["do X"])
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", instructions=["do X"])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_instructions(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="my response output",
            instructions=["first instruction", "second instruction"],
        )
        last_call = judge.calls[-1]
        assert "my response output" in last_call["prompt"]
        assert "first instruction" in last_call["prompt"]
        assert "second instruction" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_single_instruction(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", instructions=["do X"])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = InstructionCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", instructions=["do X"])
        assert result.latency_ms >= 0
