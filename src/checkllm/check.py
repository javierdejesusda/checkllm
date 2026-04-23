from __future__ import annotations

import asyncio
import logging
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from checkllm.cache import JudgeCache, _cache_key
from checkllm.check_deterministic import DeterministicChecksMixin
from checkllm.check_judge import JudgeChecksMixin
from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend, JudgeConfigError
from checkllm.logging_config import setup_logging
from checkllm.models import CheckFailedError, CheckResult
from checkllm.pricing import build_cost_breakdown, infer_provider
from checkllm.progress import (
    emit_check_completed,
    emit_test_completed,
    emit_test_started,
)

if TYPE_CHECKING:
    from checkllm.expect import SoftCheckProxy

logger = logging.getLogger("checkllm.check")


class CheckCollector(DeterministicChecksMixin, JudgeChecksMixin):
    """Collects check results during a test and raises on teardown if any failed.

    This is the core object exposed as the ``check`` pytest fixture.
    """

    def __init__(
        self,
        config: CheckllmConfig,
        judge: JudgeBackend | None = None,
        test_id: str = "",
    ) -> None:
        self.config = config
        self.results: list[CheckResult] = []
        self._deterministic = DeterministicChecks()
        self._judge = judge
        self._accumulated_cost: float = 0.0
        self._skipped_budget: int = 0
        self._test_id: str = test_id
        self._test_started_at: float | None = None

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

        # Soft assertions proxy (lazy init)
        self._expect: SoftCheckProxy | None = None

    @property
    def expect(self) -> SoftCheckProxy:
        """Soft assertions -- same API as check but never fail the test.

        Usage::

            check.expect.contains(output, "maybe present")  # won't fail
            check.expect.relevance(output, query="...")      # recorded, not enforced
        """
        if self._expect is None:
            from checkllm.expect import SoftCheckProxy

            self._expect = SoftCheckProxy(self)
        return self._expect

    def that(self, output: str):
        """Start a fluent assertion chain for the given output.

        Usage::

            check.that(output).contains("Python").has_no_pii().max_tokens(200)
        """
        from checkllm.chain import AssertionChain

        return AssertionChain(self, output)

    def _get_judge(self) -> JudgeBackend:
        if self._judge is None:
            backend = self.config.judge_backend
            model = self.config.judge_model

            if backend == "auto":
                from checkllm.discovery import (
                    detect_judge_backend,
                    format_no_judge_error,
                )

                detected = detect_judge_backend()
                if detected is None:
                    raise JudgeConfigError(format_no_judge_error())
                detected_backend, detected_model = detected
                backend = detected_backend
                # Only use detected model if user hasn't explicitly configured one
                if model == "gpt-4o":  # default value means not explicitly set
                    model = detected_model

            if backend == "anthropic":
                from checkllm.judge import AnthropicJudge

                self._judge = AnthropicJudge(model=model)
            elif backend == "gemini":
                from checkllm.providers import GeminiJudge

                self._judge = GeminiJudge(model=model)
            elif backend == "ollama":
                from checkllm.providers import OllamaJudge

                self._judge = OllamaJudge(model=model)
            elif backend == "litellm":
                from checkllm.providers import LiteLLMJudge

                self._judge = LiteLLMJudge(model=model)
            elif backend == "azure":
                from checkllm.providers import AzureOpenAIJudge

                self._judge = AzureOpenAIJudge(deployment=model)
            else:
                from checkllm.judge import OpenAIJudge

                self._judge = OpenAIJudge(model=model)
        return self._judge

    def bind_test(self, test_id: str) -> None:
        """Associate the collector with a test identifier and emit a start event.

        The pytest plugin calls this from its fixture so live dashboards can
        correlate every check with the originating test.  Safe to call more
        than once -- subsequent calls update the ``test_id`` without
        re-emitting ``test_started``.

        Args:
            test_id: Pytest node id or caller-supplied identifier.
        """
        first = not self._test_id
        self._test_id = test_id
        if first and test_id:
            self._test_started_at = time.time()
            try:
                emit_test_started(test_id=test_id)
            except Exception:  # noqa: BLE001
                logger.debug("progress broker emit_test_started failed", exc_info=True)

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

    def _cached_judge_check(
        self,
        metric_name: str,
        metric_factory,
        coro_factory,
        cache_kwargs: dict,
        runs: int | None = None,
        threshold: float | None = None,
        input_preview: str | None = None,
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

        # Attach diagnostic context
        if threshold is not None and result.threshold is None:
            result.threshold = threshold
        if input_preview is not None and result.input_preview is None:
            result.input_preview = input_preview[:200]

        # Attach cost breakdown using the judge's last token counts.
        if result.cost_breakdown is None:
            breakdown = build_cost_breakdown(
                model=model,
                input_tokens=int(getattr(judge, "last_input_tokens", 0) or 0),
                output_tokens=int(getattr(judge, "last_output_tokens", 0) or 0),
                metric=metric_name,
                test_id=self._test_id,
            )
            result.cost_breakdown = breakdown.to_dict()

        # Store in cache
        self._cache.put(key, metric_name, model, result)

        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def _fire_before_hook(self, metric_name: str, kwargs: dict) -> dict:
        """Fire the before_check hook, returning potentially modified kwargs."""
        from checkllm.hookspecs import plugin_manager

        pm = plugin_manager()
        results = pm.hook.checkllm_before_check(
            metric_name=metric_name,
            kwargs=kwargs,
        )
        for r in results:
            if r is not None:
                return r
        return kwargs

    def _emit_check_event(self, result: CheckResult) -> None:
        """Fire a ``check_completed`` progress event for a result."""
        if not self._test_id:
            return
        breakdown = result.cost_breakdown or {}
        provider = str(breakdown.get("provider", "")) if isinstance(breakdown, dict) else ""
        model = str(breakdown.get("model", "")) if isinstance(breakdown, dict) else ""
        try:
            emit_check_completed(
                test_id=self._test_id,
                metric=result.metric_name,
                passed=result.passed,
                score=float(result.score),
                cost=float(result.cost),
                duration_ms=float(result.latency_ms),
                provider=provider or infer_provider(model),
                model=model,
            )
        except Exception:  # noqa: BLE001
            logger.debug("progress broker emit_check_completed failed", exc_info=True)

    def _fire_after_hook(self, result) -> None:
        """Fire after_check and on_failure hooks."""
        from checkllm.hookspecs import plugin_manager

        pm = plugin_manager()
        pm.hook.checkllm_after_check(
            result=result,
            metric_name=result.metric_name,
        )
        if not result.passed:
            pm.hook.checkllm_on_failure(
                result=result,
                metric_name=result.metric_name,
            )
        # Publish to the live dashboard broker (no-op if no subscribers).
        self._emit_check_event(result)

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
        # Emit a test_completed event before raising so subscribers observe
        # both passing and failing tests consistently.
        if self._test_id:
            duration_ms = 0.0
            if self._test_started_at is not None:
                duration_ms = max(0.0, (time.time() - self._test_started_at) * 1000.0)
            try:
                emit_test_completed(
                    test_id=self._test_id,
                    passed=not failed,
                    duration_ms=duration_ms,
                    checks=len(self.results),
                    cost=float(self._accumulated_cost),
                )
            except Exception:  # noqa: BLE001
                logger.debug("progress broker emit_test_completed failed", exc_info=True)

        if failed:
            raise CheckFailedError(self.results)

    @property
    def total_cost(self) -> float:
        """Total cost accumulated across all judge calls this session."""
        return self._accumulated_cost

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats()

    def __repr__(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_cost = sum(r.cost for r in self.results)
        return (
            f"CheckCollector(checks={len(self.results)}, "
            f"passed={passed}, failed={failed}, cost=${total_cost:.4f})"
        )
