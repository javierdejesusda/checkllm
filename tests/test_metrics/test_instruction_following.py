from __future__ import annotations

import pytest

from checkllm.metrics.instruction_following import InstructionFollowingMetric
from checkllm.testing import MockJudge


class TestInstructionFollowingMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_following(self, judge):
        judge.set_default(score=0.95, reasoning="Output follows all instructions")
        metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output='{"name": "Python", "type": "language"}',
            instructions="Respond in JSON format with 'name' and 'type' fields.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "instruction_following"

    @pytest.mark.asyncio
    async def test_fails_when_not_following(self, judge):
        judge.set_default(score=0.2, reasoning="Output ignores format instructions")
        metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a programming language.",
            instructions="Respond in JSON format with 'name' and 'type' fields.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", instructions="test")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", instructions="test")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially following")
        metric = InstructionFollowingMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", instructions="test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom instruction prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test", instructions="test")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom instruction prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test", instructions="test")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_instructions(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = InstructionFollowingMetric(judge=judge, threshold=0.8)
        await metric.evaluate(output="my output", instructions="my instructions")
        last_call = judge.calls[-1]
        assert "my output" in last_call["prompt"]
        assert "my instructions" in last_call["prompt"]
