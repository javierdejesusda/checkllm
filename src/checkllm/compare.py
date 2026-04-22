"""Provider matrix testing — run prompts against multiple LLM providers and compare results.

Usage::

    from checkllm.compare import ProviderMatrix, MatrixResult

    providers = {"gpt4": gpt4_judge, "claude": claude_judge}
    matrix = ProviderMatrix(prompts=["Explain gravity"], providers=providers)
    result = matrix.run()
    print(result.best_provider("score"))
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from checkllm.models import JudgeResponse


class ComparisonResult(BaseModel):
    """Result for one prompt and provider combination."""

    provider: str
    prompt: str
    output: str
    latency_ms: int = 0
    cost: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    reasoning: dict[str, str] = Field(default_factory=dict)


class MatrixResult(BaseModel):
    """Aggregated results across all prompts and providers."""

    results: list[ComparisonResult] = Field(default_factory=list)

    def best_provider(self, metric: str) -> str:
        """Return the provider with the highest average score for a metric.

        Args:
            metric: The score key to rank providers by.

        Returns:
            The provider name with the highest average score for the metric.

        Raises:
            ValueError: If no results contain the requested metric.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            if metric in result.scores:
                totals[result.provider] = totals.get(result.provider, 0.0) + result.scores[metric]
                counts[result.provider] = counts.get(result.provider, 0) + 1

        if not totals:
            raise ValueError(f"No results contain metric '{metric}'")

        averages = {p: totals[p] / counts[p] for p in totals}
        return max(averages, key=lambda p: averages[p])

    def cheapest_provider(self) -> str:
        """Return the provider with the lowest average cost per prompt.

        Returns:
            The provider name with the lowest average cost.

        Raises:
            ValueError: If there are no results.
        """
        if not self.results:
            raise ValueError("No results available")

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            totals[result.provider] = totals.get(result.provider, 0.0) + result.cost
            counts[result.provider] = counts.get(result.provider, 0) + 1

        averages = {p: totals[p] / counts[p] for p in totals}
        return min(averages, key=lambda p: averages[p])

    def fastest_provider(self) -> str:
        """Return the provider with the lowest average latency per prompt.

        Returns:
            The provider name with the lowest average latency in milliseconds.

        Raises:
            ValueError: If there are no results.
        """
        if not self.results:
            raise ValueError("No results available")

        totals: dict[str, int] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            totals[result.provider] = totals.get(result.provider, 0) + result.latency_ms
            counts[result.provider] = counts.get(result.provider, 0) + 1

        averages = {p: totals[p] / counts[p] for p in totals}
        return min(averages, key=lambda p: averages[p])

    def summary(self) -> str:
        """Return a human-readable summary table of per-provider averages.

        Returns:
            A multi-line string with provider, avg latency, avg cost, and scores.
        """
        if not self.results:
            return "No results."

        providers = sorted({r.provider for r in self.results})
        lines = ["Provider Matrix Summary", "=" * 50]

        for provider in providers:
            provider_results = [r for r in self.results if r.provider == provider]
            avg_latency = sum(r.latency_ms for r in provider_results) / len(provider_results)
            avg_cost = sum(r.cost for r in provider_results) / len(provider_results)

            all_metric_keys: set[str] = set()
            for r in provider_results:
                all_metric_keys.update(r.scores.keys())

            score_parts: list[str] = []
            for key in sorted(all_metric_keys):
                values = [r.scores[key] for r in provider_results if key in r.scores]
                if values:
                    avg = sum(values) / len(values)
                    score_parts.append(f"{key}={avg:.2f}")

            scores_str = ", ".join(score_parts) if score_parts else "n/a"
            lines.append(
                f"{provider}: latency={avg_latency:.0f}ms  cost=${avg_cost:.4f}  scores=[{scores_str}]"
            )

        return "\n".join(lines)


class ProviderMatrix:
    """Run a set of prompts against multiple providers and collect comparison results.

    Args:
        prompts: List of prompt strings to evaluate.
        providers: Mapping of provider name to JudgeBackend instance.
        metrics: Optional mapping of metric name to JudgeBackend used to score outputs.
    """

    def __init__(
        self,
        prompts: list[str],
        providers: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.prompts = prompts
        self.providers = providers
        self.metrics = metrics or {}

    async def arun(self) -> MatrixResult:
        """Run all prompt/provider combinations asynchronously.

        Returns:
            A MatrixResult containing all ComparisonResult entries.
        """
        results: list[ComparisonResult] = []

        for prompt in self.prompts:
            for provider_name, backend in self.providers.items():
                start = time.monotonic()
                response: JudgeResponse = await backend.evaluate(prompt=prompt)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                scores: dict[str, float] = {}
                reasoning: dict[str, str] = {}

                for metric_name, judge in self.metrics.items():
                    judge_response: JudgeResponse = await judge.evaluate(
                        prompt=response.raw_output or response.reasoning
                    )
                    scores[metric_name] = judge_response.score
                    reasoning[metric_name] = judge_response.reasoning

                results.append(
                    ComparisonResult(
                        provider=provider_name,
                        prompt=prompt,
                        output=response.raw_output or response.reasoning,
                        latency_ms=elapsed_ms,
                        cost=response.cost,
                        scores=scores,
                        reasoning=reasoning,
                    )
                )

        return MatrixResult(results=results)

    def run(self) -> MatrixResult:
        """Run all prompt/provider combinations synchronously.

        Returns:
            A MatrixResult containing all ComparisonResult entries.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.arun()).result()

        return asyncio.run(self.arun())
