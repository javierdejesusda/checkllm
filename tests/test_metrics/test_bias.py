from __future__ import annotations

import pytest

from checkllm.metrics.bias import BiasMetric
from checkllm.testing import MockJudge


class TestBiasMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_unbiased(self, judge):
        judge.set_default(score=0.95, reasoning="No bias detected")
        metric = BiasMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="All candidates are evaluated based on their qualifications and experience.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "bias"

    @pytest.mark.asyncio
    async def test_fails_when_biased(self, judge):
        judge.set_default(score=0.2, reasoning="Contains gender stereotypes")
        metric = BiasMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Women are naturally better at caregiving while men are better leaders.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = BiasMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Some bias present")
        metric = BiasMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = BiasMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom bias prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom bias prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = BiasMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_optional_categories(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = BiasMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="test output",
            categories=["gender", "race"],
        )
        assert result.passed is True
        last_call = judge.calls[-1]
        assert "gender" in last_call["prompt"]
        assert "race" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_without_categories(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = BiasMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test output")
        assert result.passed is True
