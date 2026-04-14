import pytest
from bench.adapters.checkllm_adapter import CheckllmAdapter
from bench.schema import BenchmarkSample, GroundTruth, MetricFamily
from checkllm.testing import MockJudge


@pytest.fixture
def sample():
    return BenchmarkSample(
        sample_id="x1",
        dataset="halubench",
        query="Who wrote Hamlet?",
        answer="William Shakespeare wrote Hamlet.",
        context="Hamlet was written by William Shakespeare around 1600.",
        ground_truth=GroundTruth(label=1.0, kind="binary"),
    )


@pytest.mark.asyncio
async def test_checkllm_adapter_returns_score_for_hallucination(sample):
    judge = MockJudge(default_score=0.92)
    adapter = CheckllmAdapter(judge=judge)
    result = await adapter.score(sample, MetricFamily.HALLUCINATION, "mock-judge")
    assert result.framework == "checkllm"
    assert result.metric_family == MetricFamily.HALLUCINATION
    assert result.metric_name == "hallucination"
    assert result.score == 0.92
    assert result.sample_id == "x1"
    assert result.judge_model == "mock-judge"


@pytest.mark.asyncio
async def test_checkllm_adapter_supports_three_families(sample):
    judge = MockJudge(default_score=0.9)
    adapter = CheckllmAdapter(judge=judge)
    for family in [
        MetricFamily.HALLUCINATION,
        MetricFamily.FAITHFULNESS,
        MetricFamily.CONTEXT_RELEVANCE,
    ]:
        assert adapter.supports(family)
        result = await adapter.score(sample, family, "mock-judge")
        assert result.metric_family is family
