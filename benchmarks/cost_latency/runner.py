"""Cost and latency benchmark runner for CheckLLM metrics.

Runs a configurable set of metrics against a small sample dataset and
records the wall-clock latency and judge cost of each call. Results are
aggregated into a :class:`BenchmarkReport` that can be serialised as
JSON or rendered as a markdown table for publication.

Usage::

    python -m benchmarks.cost_latency.runner \\
        --metrics faithfulness,hallucination,toxicity \\
        --judge stub \\
        --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult, JudgeResponse


class StubJudge:
    """Deterministic in-process judge for benchmark dry-runs.

    The stub returns a fixed score and a tiny synthetic cost so the
    benchmark harness can be exercised without hitting a live API. It
    intentionally sleeps a small amount to simulate realistic latency
    distributions.
    """

    def __init__(
        self,
        score: float = 0.9,
        cost_per_call: float = 0.0012,
        latency_ms: float = 5.0,
    ) -> None:
        self.score = score
        self.cost_per_call = cost_per_call
        self.latency_ms = latency_ms
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        await asyncio.sleep(self.latency_ms / 1000.0)
        self.last_cost = self.cost_per_call
        self.total_cost += self.cost_per_call
        return JudgeResponse(
            score=self.score,
            reasoning="stub judge",
            raw_output='{"score": %.2f, "reasoning": "stub"}' % self.score,
            cost=self.cost_per_call,
        )


DEFAULT_SAMPLE_INPUTS: tuple[str, ...] = (
    "The capital of France is Paris, which is located on the Seine river.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Quantum computers use qubits that can be in superposition of 0 and 1.",
    "The Great Wall of China was built over many dynasties for defense.",
    "Gradient descent is an iterative optimization algorithm for differentiable functions.",
)


@dataclass
class MetricBenchmark:
    """Aggregated measurements for a single metric.

    Attributes:
        metric: Metric identifier.
        runs: Number of successful invocations contributing to the stats.
        mean_latency_ms: Arithmetic mean wall-clock latency.
        p50_latency_ms: Median latency.
        p95_latency_ms: 95th-percentile latency.
        mean_cost_usd: Mean cost per invocation.
        total_cost_usd: Sum of costs over all invocations.
        errors: Number of invocations that raised an exception.
    """

    metric: str
    runs: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    total_cost_usd: float
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "runs": self.runs,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_cost_usd": round(self.mean_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "errors": self.errors,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark result.

    Attributes:
        judge_model: Identifier of the judge backend used.
        sample_count: Number of distinct sample inputs per metric.
        runs_per_metric: Number of repetitions per (metric, sample).
        results: Per-metric aggregates.
        generated_at: ISO-8601-ish timestamp at report creation time.
    """

    judge_model: str
    sample_count: int
    runs_per_metric: int
    results: list[MetricBenchmark] = field(default_factory=list)
    generated_at: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the report."""
        return {
            "judge_model": self.judge_model,
            "sample_count": self.sample_count,
            "runs_per_metric": self.runs_per_metric,
            "generated_at": self.generated_at,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown_table(self) -> str:
        """Render the report as a GitHub-flavored markdown table."""
        header = (
            "| Metric | Mean latency (ms) | p95 latency (ms) | Mean cost ($) | Notes |\n"
            "|---|---:|---:|---:|---|"
        )
        rows: list[str] = [header]
        for r in self.results:
            note = "errors: %d" % r.errors if r.errors else ""
            rows.append(
                "| {metric} | {mean:.1f} | {p95:.1f} | {cost:.6f} | {note} |".format(
                    metric=r.metric,
                    mean=r.mean_latency_ms,
                    p95=r.p95_latency_ms,
                    cost=r.mean_cost_usd,
                    note=note,
                )
            )
        return "\n".join(rows)


_METRIC_CALLERS: dict[str, Any] = {}


def _register(name: str):
    """Decorator: register a metric caller under *name*."""

    def decorator(fn: Any) -> Any:
        _METRIC_CALLERS[name] = fn
        return fn

    return decorator


@_register("faithfulness")
async def _call_faithfulness(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.faithfulness import FaithfulnessMetric

    metric = FaithfulnessMetric(judge=judge)
    return await metric.evaluate(output=sample, context=sample)


@_register("hallucination")
async def _call_hallucination(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.hallucination import HallucinationMetric

    metric = HallucinationMetric(judge=judge)
    return await metric.evaluate(output=sample, context=sample)


@_register("relevance")
async def _call_relevance(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.relevance import RelevanceMetric

    metric = RelevanceMetric(judge=judge)
    return await metric.evaluate(output=sample, query=sample)


@_register("toxicity")
async def _call_toxicity(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.toxicity import ToxicityMetric

    metric = ToxicityMetric(judge=judge)
    return await metric.evaluate(output=sample)


@_register("bias")
async def _call_bias(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.bias import BiasMetric

    metric = BiasMetric(judge=judge)
    return await metric.evaluate(output=sample)


@_register("coherence")
async def _call_coherence(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.coherence import CoherenceMetric

    metric = CoherenceMetric(judge=judge)
    return await metric.evaluate(output=sample)


@_register("correctness")
async def _call_correctness(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.correctness import CorrectnessMetric

    metric = CorrectnessMetric(judge=judge)
    return await metric.evaluate(output=sample, expected=sample)


@_register("rubric")
async def _call_rubric(judge: JudgeBackend, sample: str) -> CheckResult:
    from checkllm.metrics.rubric import RubricMetric

    metric = RubricMetric(judge=judge)
    return await metric.evaluate(output=sample, criteria="Output should be clear and factual.")


async def _generic_model_call(judge: JudgeBackend, sample: str, name: str) -> CheckResult:
    """Fallback caller for metrics without a bespoke entry point.

    The benchmark only measures latency and cost of a single judge
    invocation, so for metrics that already wrap exactly one judge call
    this is a good stand-in. It returns a synthetic CheckResult whose
    cost reflects the judge's last_cost.
    """
    start = time.perf_counter_ns()
    response = await judge.evaluate(
        prompt=(
            f"Evaluate the following output for the '{name}' metric.\n"
            f"Output: {sample}\n"
            "Score 0.0 to 1.0."
        ),
        system_prompt=(
            "You are an expert evaluator. Respond with JSON only: "
            '{"score": <float 0-1>, "reasoning": "<explanation>"}.'
        ),
    )
    elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
    return CheckResult(
        passed=response.score >= 0.5,
        score=response.score,
        reasoning=response.reasoning,
        cost=response.cost,
        latency_ms=int(elapsed_ms),
        metric_name=name,
    )


class CostLatencyBenchmark:
    """Measure cost and latency of a batch of metrics.

    Attributes:
        metrics: Names of metrics to benchmark.
        judge: Judge backend passed to each metric.
        sample_inputs: Input strings used as output/context/query.
        runs_per_metric: Number of repetitions per (metric, sample).
    """

    def __init__(
        self,
        metrics: Sequence[str],
        judge: JudgeBackend,
        sample_inputs: Sequence[str] | None = None,
        runs_per_metric: int = 5,
    ) -> None:
        self.metrics = list(metrics)
        self.judge = judge
        self.sample_inputs = list(sample_inputs) if sample_inputs else list(DEFAULT_SAMPLE_INPUTS)
        self.runs_per_metric = runs_per_metric

    async def _run_one(self, metric_name: str, sample: str) -> tuple[float, float, bool]:
        """Run a single metric invocation and return (latency_ms, cost, error_flag)."""
        caller = _METRIC_CALLERS.get(metric_name)
        start = time.perf_counter_ns()
        try:
            if caller is not None:
                result = await caller(self.judge, sample)
            else:
                result = await _generic_model_call(self.judge, sample, metric_name)
        except Exception:
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            return elapsed_ms, 0.0, True
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return elapsed_ms, float(getattr(result, "cost", 0.0) or 0.0), False

    async def run(self) -> BenchmarkReport:
        """Execute the benchmark and return an aggregated report."""
        results: list[MetricBenchmark] = []
        for metric_name in self.metrics:
            latencies: list[float] = []
            costs: list[float] = []
            errors = 0
            for sample in self.sample_inputs:
                for _ in range(self.runs_per_metric):
                    lat, cost, err = await self._run_one(metric_name, sample)
                    if err:
                        errors += 1
                        continue
                    latencies.append(lat)
                    costs.append(cost)

            if latencies:
                mean_lat = statistics.fmean(latencies)
                sorted_lat = sorted(latencies)
                p50 = statistics.median(sorted_lat)
                p95_index = max(0, int(round(0.95 * (len(sorted_lat) - 1))))
                p95 = sorted_lat[p95_index]
                mean_cost = statistics.fmean(costs) if costs else 0.0
                total_cost = sum(costs)
            else:
                mean_lat = p50 = p95 = 0.0
                mean_cost = 0.0
                total_cost = 0.0

            results.append(
                MetricBenchmark(
                    metric=metric_name,
                    runs=len(latencies),
                    mean_latency_ms=mean_lat,
                    p50_latency_ms=p50,
                    p95_latency_ms=p95,
                    mean_cost_usd=mean_cost,
                    total_cost_usd=total_cost,
                    errors=errors,
                )
            )

        return BenchmarkReport(
            judge_model=getattr(self.judge, "model", type(self.judge).__name__),
            sample_count=len(self.sample_inputs),
            runs_per_metric=self.runs_per_metric,
            results=results,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


def _create_judge(name: str, model: str | None) -> JudgeBackend:
    """Factory for CLI ``--judge`` values.

    Args:
        name: Either ``"stub"`` or a checkllm provider identifier.
        model: Optional model override for non-stub judges.

    Returns:
        A configured judge backend.

    Raises:
        ValueError: If the judge cannot be created (e.g. missing API
            key for a live provider).
    """
    if name == "stub":
        return StubJudge()
    from checkllm.providers import create_judge

    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    return create_judge(name, **kwargs)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run cost/latency benchmarks for CheckLLM metrics."
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,hallucination,relevance,toxicity,rubric",
        help="Comma-separated list of metric names.",
    )
    parser.add_argument(
        "--judge",
        default="stub",
        help="Judge backend id. 'stub' runs locally with no API calls.",
    )
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--runs", type=int, default=5, help="Runs per metric and sample.")
    parser.add_argument(
        "--samples",
        type=int,
        default=len(DEFAULT_SAMPLE_INPUTS),
        help="Number of sample inputs to use from the built-in set.",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format written to stdout.",
    )
    args = parser.parse_args(argv)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    samples = list(DEFAULT_SAMPLE_INPUTS[: max(1, args.samples)])
    judge = _create_judge(args.judge, args.model)

    bench = CostLatencyBenchmark(
        metrics=metrics,
        judge=judge,
        sample_inputs=samples,
        runs_per_metric=args.runs,
    )
    report = asyncio.run(bench.run())

    if args.output == "json":
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(report.to_markdown_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BenchmarkReport",
    "CostLatencyBenchmark",
    "DEFAULT_SAMPLE_INPUTS",
    "MetricBenchmark",
    "StubJudge",
]
