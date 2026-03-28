from __future__ import annotations

import asyncio
import statistics
from typing import Any, Type

from pydantic import BaseModel

from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend, JudgeConfigError, OpenAIJudge
from checkllm.metrics.hallucination import HallucinationMetric
from checkllm.metrics.relevance import RelevanceMetric
from checkllm.metrics.rubric import RubricMetric
from checkllm.metrics.toxicity import ToxicityMetric
from checkllm.models import CheckFailedError, CheckResult


class CheckCollector:
    """Collects check results during a test and raises on teardown if any failed.

    This is the core object exposed as the ``check`` pytest fixture.
    """

    def __init__(
        self,
        config: CheckllmConfig,
        judge: JudgeBackend | None = None,
    ) -> None:
        self.config = config
        self.results: list[CheckResult] = []
        self._deterministic = DeterministicChecks()
        self._judge = judge

    def _get_judge(self) -> JudgeBackend:
        if self._judge is None:
            if self.config.judge_backend == "anthropic":
                from checkllm.judge import AnthropicJudge
                self._judge = AnthropicJudge(model=self.config.judge_model)
            else:
                self._judge = OpenAIJudge(model=self.config.judge_model)
        return self._judge

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _run_with_repeats(self, coro_factory, runs: int | None = None):
        """Run an async check N times and return the aggregated result."""
        n = runs if runs is not None else self.config.runs_per_test
        if n <= 1:
            return self._run_async(coro_factory())

        results = []
        for _ in range(n):
            results.append(self._run_async(coro_factory()))

        scores = [r.score for r in results]
        avg_score = statistics.mean(scores)
        total_cost = sum(r.cost for r in results)
        total_latency = sum(r.latency_ms for r in results)
        pass_rate = sum(1 for r in results if r.passed) / len(results)

        return CheckResult(
            passed=pass_rate >= 0.5,
            score=avg_score,
            reasoning=f"Aggregated over {n} runs (pass rate: {pass_rate:.0%}, scores: {', '.join(f'{s:.2f}' for s in scores)})",
            cost=total_cost,
            latency_ms=total_latency,
            metric_name=results[0].metric_name,
        )

    # --- Deterministic checks ---

    def contains(self, output: str, substring: str) -> CheckResult:
        result = self._deterministic.contains(output, substring)
        self.results.append(result)
        return result

    def not_contains(self, output: str, substring: str) -> CheckResult:
        result = self._deterministic.not_contains(output, substring)
        self.results.append(result)
        return result

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        result = self._deterministic.max_tokens(output, limit)
        self.results.append(result)
        return result

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        result = self._deterministic.latency(actual_ms, max_ms)
        self.results.append(result)
        return result

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        result = self._deterministic.cost(actual_usd, max_usd)
        self.results.append(result)
        return result

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        result = self._deterministic.json_schema(output, schema)
        self.results.append(result)
        return result

    def regex(self, output: str, pattern: str) -> CheckResult:
        result = self._deterministic.regex(output, pattern)
        self.results.append(result)
        return result

    def exact_match(self, output: str, expected: str, ignore_case: bool = False) -> CheckResult:
        result = self._deterministic.exact_match(output, expected, ignore_case)
        self.results.append(result)
        return result

    def starts_with(self, output: str, prefix: str) -> CheckResult:
        result = self._deterministic.starts_with(output, prefix)
        self.results.append(result)
        return result

    def ends_with(self, output: str, suffix: str) -> CheckResult:
        result = self._deterministic.ends_with(output, suffix)
        self.results.append(result)
        return result

    # --- LLM-as-judge checks ---

    def hallucination(
        self, output: str, context: str, threshold: float | None = None, runs: int | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        result = self._run_with_repeats(
            lambda: metric.evaluate(output=output, context=context), runs
        )
        self.results.append(result)
        return result

    def relevance(
        self, output: str, query: str, threshold: float | None = None, runs: int | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        result = self._run_with_repeats(
            lambda: metric.evaluate(output=output, query=query), runs
        )
        self.results.append(result)
        return result

    def toxicity(
        self, output: str, threshold: float | None = None, runs: int | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        result = self._run_with_repeats(
            lambda: metric.evaluate(output=output), runs
        )
        self.results.append(result)
        return result

    def rubric(
        self, output: str, criteria: str, threshold: float | None = None, runs: int | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        result = self._run_with_repeats(
            lambda: metric.evaluate(output=output, criteria=criteria, threshold=t), runs
        )
        self.results.append(result)
        return result

    # --- Custom metric support ---

    def run_metric(self, name: str, output: str, **kwargs: Any) -> CheckResult:
        """Run a custom registered metric by name.

        Usage::

            @checkllm.metric("my_metric")
            def my_metric(output: str, **kwargs) -> CheckResult:
                ...

            def test_it(check):
                check.run_metric("my_metric", output="hello", custom_arg="value")
        """
        from checkllm.metrics import _global_registry

        if name not in _global_registry.metrics:
            raise ValueError(
                f"Metric '{name}' is not registered. "
                f"Available: {', '.join(_global_registry.list_metrics()) or '(none)'}"
            )
        func = _global_registry.metrics[name]
        result = func(output, **kwargs)
        self.results.append(result)
        return result

    # --- Async LLM-as-judge checks ---

    async def ahallucination(
        self, output: str, context: str, threshold: float | None = None,
    ) -> CheckResult:
        """Async version of hallucination check."""
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        result = await metric.evaluate(output=output, context=context)
        self.results.append(result)
        return result

    async def arelevance(
        self, output: str, query: str, threshold: float | None = None,
    ) -> CheckResult:
        """Async version of relevance check."""
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        result = await metric.evaluate(output=output, query=query)
        self.results.append(result)
        return result

    async def atoxicity(
        self, output: str, threshold: float | None = None,
    ) -> CheckResult:
        """Async version of toxicity check."""
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        result = await metric.evaluate(output=output)
        self.results.append(result)
        return result

    async def arubric(
        self, output: str, criteria: str, threshold: float | None = None,
    ) -> CheckResult:
        """Async version of rubric check."""
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        result = await metric.evaluate(output=output, criteria=criteria, threshold=t)
        self.results.append(result)
        return result

    # --- Teardown ---

    def teardown(self) -> None:
        """Raise CheckFailedError if any checks failed."""
        failed = [r for r in self.results if not r.passed]
        if failed:
            raise CheckFailedError(self.results)

    # --- Repr ---

    def __repr__(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_cost = sum(r.cost for r in self.results)
        return (
            f"CheckCollector(checks={len(self.results)}, "
            f"passed={passed}, failed={failed}, cost=${total_cost:.4f})"
        )
