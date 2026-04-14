from __future__ import annotations

import asyncio
from dataclasses import dataclass

from bench.schema import BenchmarkSample, BenchmarkScore, FrameworkAdapter, MetricFamily


@dataclass
class RunSpec:
    """Specification for a benchmark run.

    Attributes:
        adapters: Framework adapters to evaluate.
        samples: Dataset samples to score.
        families: Metric families to exercise.
        judge_model: Identifier of the judge model used for scoring.
    """

    adapters: list[FrameworkAdapter]
    samples: list[BenchmarkSample]
    families: list[MetricFamily]
    judge_model: str


class BenchmarkRunner:
    """Async runner that evaluates adapters against samples with concurrency and budget caps.

    Args:
        max_concurrency: Maximum number of concurrent scoring tasks.
        budget_usd: Maximum total spend in USD before halting new tasks.
    """

    def __init__(self, max_concurrency: int = 8, budget_usd: float = 50.0) -> None:
        self.max_concurrency = max_concurrency
        self.budget_usd = budget_usd
        self._spent: float = 0.0
        self._spent_lock = asyncio.Lock()

    async def _score_one(
        self,
        adapter: FrameworkAdapter,
        sample: BenchmarkSample,
        family: MetricFamily,
        judge_model: str,
        sem: asyncio.Semaphore,
    ) -> BenchmarkScore | None:
        """Score a single (adapter, sample, family) triple with budget enforcement.

        Args:
            adapter: The framework adapter to invoke.
            sample: The benchmark sample to evaluate.
            family: The metric family to score.
            judge_model: Identifier of the judge model.
            sem: Semaphore controlling max concurrency.

        Returns:
            A BenchmarkScore, or None if the budget was already exhausted.
        """
        async with self._spent_lock:
            if self._spent >= self.budget_usd:
                return None
        async with sem:
            try:
                result = await adapter.score(sample, family, judge_model)
            except Exception as err:
                return BenchmarkScore(
                    framework=getattr(adapter, "framework", "unknown"),
                    dataset=sample.dataset,
                    metric_family=family,
                    metric_name="error",
                    sample_id=sample.sample_id,
                    score=0.0,
                    passed=False,
                    latency_ms=0,
                    cost_usd=0.0,
                    judge_model=judge_model,
                    reasoning=f"adapter error: {err}",
                )
            async with self._spent_lock:
                self._spent += result.cost_usd
            return result

    async def run(self, spec: RunSpec) -> list[BenchmarkScore]:
        """Execute all scoring tasks defined by the run spec.

        Tasks are dispatched concurrently up to max_concurrency. No new task
        starts once the cumulative spend reaches budget_usd.

        Args:
            spec: The run specification describing adapters, samples, and families.

        Returns:
            A list of BenchmarkScore objects for every completed (non-budgeted) task.
        """
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks: list[asyncio.Task] = []
        for adapter in spec.adapters:
            for family in spec.families:
                if not adapter.supports(family):
                    continue
                for sample in spec.samples:
                    tasks.append(
                        asyncio.create_task(
                            self._score_one(adapter, sample, family, spec.judge_model, sem)
                        )
                    )
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
