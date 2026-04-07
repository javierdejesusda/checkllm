from __future__ import annotations

import pytest

from checkllm.metrics.citation_accuracy import CitationAccuracyMetric
from checkllm.testing import MockJudge


class TestCitationAccuracyMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_citations_accurate(self, judge):
        judge.set_default(score=0.95, reasoning="All citations accurately reference sources")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output=(
                "According to [Source 1], Python was created in 1991. "
                "As noted in [Source 2], it supports multiple paradigms."
            ),
            sources=[
                "Python was created by Guido van Rossum in 1991.",
                "Python supports object-oriented, procedural, and functional programming.",
            ],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "citation_accuracy"

    @pytest.mark.asyncio
    async def test_fails_when_citations_inaccurate(self, judge):
        judge.set_default(score=0.2, reasoning="Citations reference wrong sources")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="According to [Source 1], Python supports multiple paradigms.",
            sources=[
                "Python was created by Guido van Rossum in 1991.",
                "Python supports object-oriented, procedural, and functional programming.",
            ],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", sources=["source1"])
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", sources=["source1"])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_sources(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="my cited output",
            sources=["first source text", "second source text"],
        )
        last_call = judge.calls[-1]
        assert "my cited output" in last_call["prompt"]
        assert "first source text" in last_call["prompt"]
        assert "second source text" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_single_source(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test [1]", sources=["only source"])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = CitationAccuracyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", sources=["source"])
        assert result.latency_ms >= 0
