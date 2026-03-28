from __future__ import annotations

import asyncio
from typing import Any, Type

from pydantic import BaseModel

from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend, OpenAIJudge
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

    # --- LLM-as-judge checks ---

    def hallucination(
        self, output: str, context: str, threshold: float | None = None
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        result = self._run_async(metric.evaluate(output=output, context=context))
        self.results.append(result)
        return result

    def relevance(
        self, output: str, query: str, threshold: float | None = None
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        result = self._run_async(metric.evaluate(output=output, query=query))
        self.results.append(result)
        return result

    def toxicity(
        self, output: str, threshold: float | None = None
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        result = self._run_async(metric.evaluate(output=output))
        self.results.append(result)
        return result

    def rubric(
        self, output: str, criteria: str, threshold: float | None = None
    ) -> CheckResult:
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        result = self._run_async(
            metric.evaluate(output=output, criteria=criteria, threshold=t)
        )
        self.results.append(result)
        return result

    # --- Teardown ---

    def teardown(self) -> None:
        """Raise CheckFailedError if any checks failed."""
        failed = [r for r in self.results if not r.passed]
        if failed:
            raise CheckFailedError(self.results)
