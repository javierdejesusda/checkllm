from __future__ import annotations

import asyncio
import logging
import statistics
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from checkllm.cache import JudgeCache, _cache_key
from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend, JudgeConfigError, OpenAIJudge
from checkllm.logging_config import setup_logging
from checkllm.metrics.hallucination import HallucinationMetric
from checkllm.metrics.relevance import RelevanceMetric
from checkllm.metrics.rubric import RubricMetric
from checkllm.metrics.toxicity import ToxicityMetric
from checkllm.models import CheckFailedError, CheckResult

logger = logging.getLogger("checkllm.check")


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
        self._accumulated_cost: float = 0.0
        self._skipped_budget: int = 0

        # Logging
        setup_logging(config.log_level)

        # Cache
        cache_path = Path(config.cache_dir) / "cache.db"
        self._cache = JudgeCache(
            db_path=cache_path,
            ttl_seconds=config.cache_ttl_seconds,
            enabled=config.cache_enabled,
        )

        # Concurrency semaphore for parallel judge calls
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    def _get_judge(self) -> JudgeBackend:
        if self._judge is None:
            if self.config.judge_backend == "anthropic":
                from checkllm.judge import AnthropicJudge
                self._judge = AnthropicJudge(model=self.config.judge_model)
            else:
                self._judge = OpenAIJudge(model=self.config.judge_model)
        return self._judge

    def _check_budget(self) -> bool:
        """Return True if we are within budget, False if budget exceeded."""
        if self.config.budget is None:
            return True
        return self._accumulated_cost < self.config.budget

    def _make_budget_skip_result(self, metric_name: str) -> CheckResult:
        """Create a skipped result when budget is exceeded."""
        self._skipped_budget += 1
        logger.warning(
            "Budget $%.2f exceeded (spent $%.4f) — skipping %s",
            self.config.budget,
            self._accumulated_cost,
            metric_name,
        )
        return CheckResult(
            passed=True,
            score=0.0,
            reasoning=f"Skipped: budget ${self.config.budget:.2f} exceeded (spent ${self._accumulated_cost:.4f})",
            cost=0.0,
            latency_ms=0,
            metric_name=metric_name,
        )

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

    def _track_cost(self, result: CheckResult) -> None:
        """Update the accumulated cost counter."""
        self._accumulated_cost += result.cost
        if result.cost > 0:
            logger.debug(
                "%s cost=$%.4f, total=$%.4f",
                result.metric_name,
                result.cost,
                self._accumulated_cost,
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

    # --- LLM-as-judge checks (with caching + budget) ---

    def _cached_judge_check(
        self,
        metric_name: str,
        metric_factory,
        coro_factory,
        cache_kwargs: dict,
        runs: int | None = None,
    ) -> CheckResult:
        """Run an LLM judge check with caching and budget enforcement."""
        # Budget gate
        if not self._check_budget():
            result = self._make_budget_skip_result(metric_name)
            self.results.append(result)
            return result

        # Cache lookup
        judge = self._get_judge()
        model = getattr(judge, "model", "unknown")
        key = _cache_key(metric_name, model, **cache_kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("Using cached result for %s", metric_name)
            self.results.append(cached)
            return cached

        # Execute
        result = self._run_with_repeats(coro_factory, runs)
        self._track_cost(result)

        # Store in cache
        self._cache.put(key, metric_name, model, result)

        self.results.append(result)
        return result

    def hallucination(
        self,
        output: str,
        context: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="hallucination",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context),
            cache_kwargs={"output": output, "context": context, "threshold": str(t)},
            runs=runs,
        )

    def relevance(
        self,
        output: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="relevance",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, query=query),
            cache_kwargs={"output": output, "query": query, "threshold": str(t)},
            runs=runs,
        )

    def toxicity(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="toxicity",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
        )

    def rubric(
        self,
        output: str,
        criteria: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="rubric",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, criteria=criteria, threshold=t),
            cache_kwargs={"output": output, "criteria": criteria, "threshold": str(t)},
            runs=runs,
        )

    # --- Parallel batch execution ---

    async def _run_judge_async(self, coro) -> CheckResult:
        """Run a single judge call with semaphore limiting."""
        async with self._semaphore:
            return await coro

    async def aflush(self, tasks: list) -> list[CheckResult]:
        """Run multiple judge coroutines in parallel (up to max_concurrency).

        Usage::

            results = await check.aflush([
                check.ahallucination(output, context=ctx),
                check.arelevance(output, query=q),
            ])
        """
        return await asyncio.gather(*tasks)

    # --- Custom metric support ---

    def run_metric(self, name: str, output: str, **kwargs: Any) -> CheckResult:
        """Run a custom registered metric by name."""
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
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        # Cache check
        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("hallucination", model, output=output, context=context, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("hallucination")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, context=context)
        self._track_cost(result)
        self._cache.put(key, "hallucination", model, result)
        self.results.append(result)
        return result

    async def arelevance(
        self, output: str, query: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("relevance", model, output=output, query=query, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("relevance")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, query=query)
        self._track_cost(result)
        self._cache.put(key, "relevance", model, result)
        self.results.append(result)
        return result

    async def atoxicity(
        self, output: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("toxicity", model, output=output, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("toxicity")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output)
        self._track_cost(result)
        self._cache.put(key, "toxicity", model, result)
        self.results.append(result)
        return result

    async def arubric(
        self, output: str, criteria: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("rubric", model, output=output, criteria=criteria, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("rubric")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, criteria=criteria, threshold=t)
        self._track_cost(result)
        self._cache.put(key, "rubric", model, result)
        self.results.append(result)
        return result

    # --- Teardown ---

    def teardown(self) -> None:
        """Raise CheckFailedError if any checks failed."""
        # Log session summary
        cache_stats = self._cache.stats()
        if cache_stats.get("session_hits", 0) or cache_stats.get("session_misses", 0):
            logger.info(
                "Cache: %d hits, %d misses, saved $%.4f",
                cache_stats["session_hits"],
                cache_stats["session_misses"],
                cache_stats["session_saved_cost"],
            )
        if self._accumulated_cost > 0:
            logger.info("Total cost: $%.4f", self._accumulated_cost)
        if self._skipped_budget > 0:
            logger.warning(
                "Budget: %d judge call(s) skipped (budget=$%.2f, spent=$%.4f)",
                self._skipped_budget,
                self.config.budget,
                self._accumulated_cost,
            )

        self._cache.close()

        failed = [r for r in self.results if not r.passed]
        if failed:
            raise CheckFailedError(self.results)

    # --- Properties ---

    @property
    def total_cost(self) -> float:
        """Total cost accumulated across all judge calls this session."""
        return self._accumulated_cost

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats()

    # --- Repr ---

    def __repr__(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_cost = sum(r.cost for r in self.results)
        return (
            f"CheckCollector(checks={len(self.results)}, "
            f"passed={passed}, failed={failed}, cost=${total_cost:.4f})"
        )
