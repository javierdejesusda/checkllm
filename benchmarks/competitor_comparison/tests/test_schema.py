import pytest
from bench.schema import (
    MetricFamily,
    BenchmarkSample,
    BenchmarkScore,
    GroundTruth,
)


def test_metric_family_has_five_members():
    names = {m.value for m in MetricFamily}
    assert names == {
        "hallucination",
        "faithfulness",
        "answer_relevancy",
        "context_relevance",
        "jailbreak_resistance",
    }


def test_benchmark_sample_requires_id_and_query():
    s = BenchmarkSample(
        sample_id="halubench-0001",
        dataset="halubench",
        query="What is the capital of France?",
        answer="Paris.",
        context="France's capital is Paris.",
        ground_truth=GroundTruth(label=1, kind="binary"),
    )
    assert s.sample_id == "halubench-0001"
    assert s.ground_truth.label == 1


def test_benchmark_score_clamps_to_unit_interval():
    score = BenchmarkScore(
        framework="checkllm",
        dataset="halubench",
        metric_family=MetricFamily.HALLUCINATION,
        metric_name="hallucination",
        sample_id="halubench-0001",
        score=0.87,
        passed=True,
        latency_ms=412,
        cost_usd=0.00041,
        judge_model="gpt-4o-mini",
    )
    assert 0.0 <= score.score <= 1.0


def test_benchmark_score_rejects_out_of_range():
    with pytest.raises(Exception):
        BenchmarkScore(
            framework="deepeval",
            dataset="halubench",
            metric_family=MetricFamily.HALLUCINATION,
            metric_name="hallucination",
            sample_id="x",
            score=1.5,
            passed=False,
            latency_ms=0,
            cost_usd=0.0,
            judge_model="gpt-4o",
        )
