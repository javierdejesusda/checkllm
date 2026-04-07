from __future__ import annotations

import pytest

from checkllm.metrics.topic_adherence import TopicAdherenceMetric
from checkllm.testing import MockJudge


class TestTopicAdherenceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_on_topic(self, judge):
        judge.set_default(score=0.95, reasoning="Response stays within allowed topics")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="You should eat more vegetables and exercise regularly.",
            allowed_topics=["health", "nutrition", "fitness"],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "topic_adherence"

    @pytest.mark.asyncio
    async def test_fails_when_off_topic(self, judge):
        judge.set_default(score=0.2, reasoning="Response discusses finance, not health")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="You should invest in index funds for long-term growth.",
            allowed_topics=["health", "nutrition", "fitness"],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", allowed_topics=["topic1"])
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", allowed_topics=["topic1"])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_topics_and_output(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="my output text",
            allowed_topics=["science", "technology"],
        )
        last_call = judge.calls[-1]
        assert "my output text" in last_call["prompt"]
        assert "science" in last_call["prompt"]
        assert "technology" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_single_topic(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", allowed_topics=["math"])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = TopicAdherenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", allowed_topics=["topic1"])
        assert result.latency_ms >= 0
