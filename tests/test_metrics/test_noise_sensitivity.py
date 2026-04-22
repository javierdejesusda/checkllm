from __future__ import annotations

import pytest

from checkllm.metrics.noise_sensitivity import NoiseSensitivityMetric
from checkllm.testing import MockJudge


class TestNoiseSensitivityMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_robust_to_noise(self, judge):
        judge.set_default(score=0.95, reasoning="Response is unaffected by noise")
        metric = NoiseSensitivityMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="The capital of France is Paris.",
            context="France is a country in Europe. Its capital is Paris.",
            noisy_context="Pizza is a popular Italian dish with many toppings.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "noise_sensitivity"

    @pytest.mark.asyncio
    async def test_fails_when_influenced_by_noise(self, judge):
        judge.set_default(score=0.2, reasoning="Response incorporated noisy claims")
        metric = NoiseSensitivityMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Paris is the capital of France and pizza originated there.",
            context="France is a country in Europe. Its capital is Paris.",
            noisy_context="Pizza is a popular Italian dish invented in Naples.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = NoiseSensitivityMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", context="clean", noisy_context="noise")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", context="clean", noisy_context="noise")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NoiseSensitivityMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="my output",
            context="clean context here",
            noisy_context="noisy context here",
        )
        last_call = judge.calls[-1]
        assert "my output" in last_call["prompt"]
        assert "clean context here" in last_call["prompt"]
        assert "noisy context here" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NoiseSensitivityMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", context="clean", noisy_context="noise")
        assert result.latency_ms >= 0
