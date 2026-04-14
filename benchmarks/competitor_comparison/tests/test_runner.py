import pytest
from bench.runner import BenchmarkRunner, RunSpec
from bench.adapters.checkllm_adapter import CheckllmAdapter
from bench.datasets import load_halubench_from_rows
from bench.schema import MetricFamily
from checkllm.testing import MockJudge


@pytest.mark.asyncio
async def test_runner_scores_all_samples_for_single_adapter(tiny_halubench):
    samples = load_halubench_from_rows(tiny_halubench)
    judge = MockJudge(default_score=0.9)
    adapter = CheckllmAdapter(judge=judge)

    runner = BenchmarkRunner(max_concurrency=2, budget_usd=100.0)
    spec = RunSpec(
        adapters=[adapter],
        samples=samples,
        families=[MetricFamily.HALLUCINATION],
        judge_model="mock-judge",
    )
    scores = await runner.run(spec)

    assert len(scores) == 3
    assert all(s.framework == "checkllm" for s in scores)
    assert all(s.metric_family is MetricFamily.HALLUCINATION for s in scores)


@pytest.mark.asyncio
async def test_runner_enforces_budget(tiny_halubench):
    samples = load_halubench_from_rows(tiny_halubench) * 10  # 30 samples

    class ExpensiveAdapter(CheckllmAdapter):
        async def score(self, sample, family, judge_model):
            result = await super().score(sample, family, judge_model)
            return result.model_copy(update={"cost_usd": 0.5})  # 0.5 USD each

    adapter = ExpensiveAdapter(judge=MockJudge(default_score=0.9))
    runner = BenchmarkRunner(max_concurrency=1, budget_usd=1.0)
    spec = RunSpec(
        adapters=[adapter],
        samples=samples,
        families=[MetricFamily.HALLUCINATION],
        judge_model="mock-judge",
    )
    scores = await runner.run(spec)
    assert len(scores) <= 3
