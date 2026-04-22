"""Tests for the cost/latency benchmark runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCHMARKS_ROOT = Path(__file__).parent.parent
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))

from benchmarks.cost_latency.runner import (  # noqa: E402
    BenchmarkReport,
    CostLatencyBenchmark,
    MetricBenchmark,
    StubJudge,
)


async def test_stub_judge_emits_cost_and_score() -> None:
    judge = StubJudge(score=0.8, cost_per_call=0.002, latency_ms=1.0)
    response = await judge.evaluate("prompt")
    assert response.score == pytest.approx(0.8)
    assert response.cost == pytest.approx(0.002)
    assert judge.total_cost == pytest.approx(0.002)
    assert judge.last_cost == pytest.approx(0.002)


async def test_benchmark_runs_and_aggregates() -> None:
    judge = StubJudge(latency_ms=1.0)
    bench = CostLatencyBenchmark(
        metrics=["faithfulness", "toxicity"],
        judge=judge,
        sample_inputs=["input one", "input two"],
        runs_per_metric=2,
    )
    report = await bench.run()
    assert isinstance(report, BenchmarkReport)
    assert len(report.results) == 2
    for result in report.results:
        assert result.runs == 4  # 2 samples x 2 runs
        assert result.mean_latency_ms > 0
        assert result.mean_cost_usd >= 0


async def test_report_markdown_table_contains_metric_names() -> None:
    judge = StubJudge(latency_ms=0.5)
    bench = CostLatencyBenchmark(
        metrics=["coherence", "bias"],
        judge=judge,
        sample_inputs=["x"],
        runs_per_metric=1,
    )
    report = await bench.run()
    table = report.to_markdown_table()
    assert "coherence" in table
    assert "bias" in table
    assert "Mean latency" in table


async def test_report_to_json_roundtrips() -> None:
    judge = StubJudge(latency_ms=0.2)
    bench = CostLatencyBenchmark(
        metrics=["rubric"],
        judge=judge,
        sample_inputs=["a", "b"],
        runs_per_metric=1,
    )
    report = await bench.run()
    as_dict = report.to_json()
    text = json.dumps(as_dict)
    reparsed = json.loads(text)
    assert reparsed["sample_count"] == 2
    assert reparsed["runs_per_metric"] == 1
    assert reparsed["results"][0]["metric"] == "rubric"


def test_metric_benchmark_to_dict_rounds() -> None:
    mb = MetricBenchmark(
        metric="x",
        runs=3,
        mean_latency_ms=1.234567,
        p50_latency_ms=1.1,
        p95_latency_ms=1.5,
        mean_cost_usd=0.0000012345,
        total_cost_usd=0.0000037,
        errors=0,
    )
    d = mb.to_dict()
    assert d["metric"] == "x"
    assert d["runs"] == 3
    assert d["mean_latency_ms"] == pytest.approx(1.235, rel=1e-3)


async def test_benchmark_handles_unknown_metric_via_generic_caller() -> None:
    judge = StubJudge(latency_ms=0.2)
    bench = CostLatencyBenchmark(
        metrics=["definitely_not_a_real_metric"],
        judge=judge,
        sample_inputs=["only one"],
        runs_per_metric=1,
    )
    report = await bench.run()
    assert report.results[0].runs == 1
    assert report.results[0].errors == 0


async def test_benchmark_zero_errors_on_stub_judge() -> None:
    judge = StubJudge(latency_ms=0.1)
    bench = CostLatencyBenchmark(
        metrics=["faithfulness"],
        judge=judge,
        sample_inputs=["s"],
        runs_per_metric=3,
    )
    report = await bench.run()
    assert report.results[0].errors == 0
    assert report.results[0].p95_latency_ms >= report.results[0].p50_latency_ms
