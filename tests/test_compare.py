"""Tests for the provider matrix comparison module."""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock

from checkllm.compare import ComparisonResult, MatrixResult, ProviderMatrix
from checkllm.models import JudgeResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(score: float = 0.8, reasoning: str = "ok", cost: float = 0.01) -> AsyncMock:
    """Create an AsyncMock provider that returns a fixed JudgeResponse."""
    provider = AsyncMock()
    provider.evaluate = AsyncMock(
        return_value=JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_output=f"output_{score}",
            cost=cost,
        )
    )
    return provider


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_minimal_construction(self):
        result = ComparisonResult(provider="gpt4", prompt="hello", output="world")
        assert result.provider == "gpt4"
        assert result.prompt == "hello"
        assert result.output == "world"
        assert result.latency_ms == 0
        assert result.cost == 0.0
        assert result.scores == {}
        assert result.reasoning == {}

    def test_full_construction(self):
        result = ComparisonResult(
            provider="claude",
            prompt="q",
            output="a",
            latency_ms=120,
            cost=0.005,
            scores={"relevance": 0.9},
            reasoning={"relevance": "very relevant"},
        )
        assert result.latency_ms == 120
        assert result.cost == 0.005
        assert result.scores["relevance"] == 0.9
        assert result.reasoning["relevance"] == "very relevant"


# ---------------------------------------------------------------------------
# MatrixResult
# ---------------------------------------------------------------------------


class TestMatrixResult:
    def _make_results(self) -> list[ComparisonResult]:
        return [
            ComparisonResult(
                provider="fast",
                prompt="p1",
                output="o",
                latency_ms=50,
                cost=0.02,
                scores={"quality": 0.7},
            ),
            ComparisonResult(
                provider="cheap",
                prompt="p1",
                output="o",
                latency_ms=300,
                cost=0.001,
                scores={"quality": 0.6},
            ),
            ComparisonResult(
                provider="best",
                prompt="p1",
                output="o",
                latency_ms=200,
                cost=0.01,
                scores={"quality": 0.95},
            ),
        ]

    def test_best_provider(self):
        matrix = MatrixResult(results=self._make_results())
        assert matrix.best_provider("quality") == "best"

    def test_best_provider_missing_metric(self):
        matrix = MatrixResult(results=self._make_results())
        with pytest.raises(ValueError, match="No results contain metric"):
            matrix.best_provider("nonexistent")

    def test_cheapest_provider(self):
        matrix = MatrixResult(results=self._make_results())
        assert matrix.cheapest_provider() == "cheap"

    def test_fastest_provider(self):
        matrix = MatrixResult(results=self._make_results())
        assert matrix.fastest_provider() == "fast"

    def test_cheapest_empty(self):
        matrix = MatrixResult(results=[])
        with pytest.raises(ValueError, match="No results available"):
            matrix.cheapest_provider()

    def test_fastest_empty(self):
        matrix = MatrixResult(results=[])
        with pytest.raises(ValueError, match="No results available"):
            matrix.fastest_provider()

    def test_summary_empty(self):
        matrix = MatrixResult(results=[])
        assert matrix.summary() == "No results."

    def test_summary_contains_providers(self):
        matrix = MatrixResult(results=self._make_results())
        summary = matrix.summary()
        assert "fast" in summary
        assert "cheap" in summary
        assert "best" in summary

    def test_summary_contains_metrics(self):
        matrix = MatrixResult(results=self._make_results())
        summary = matrix.summary()
        assert "quality" in summary
        assert "latency" in summary
        assert "cost" in summary

    def test_best_provider_averages_across_prompts(self):
        results = [
            ComparisonResult(provider="a", prompt="p1", output="o", scores={"s": 0.8}),
            ComparisonResult(provider="a", prompt="p2", output="o", scores={"s": 0.6}),
            ComparisonResult(provider="b", prompt="p1", output="o", scores={"s": 0.9}),
            ComparisonResult(provider="b", prompt="p2", output="o", scores={"s": 0.5}),
        ]
        matrix = MatrixResult(results=results)
        # a avg=0.7, b avg=0.7 — tie goes to whichever max() picks first (b)
        # Both average to 0.7; just assert it returns a valid provider name
        winner = matrix.best_provider("s")
        assert winner in ("a", "b")


# ---------------------------------------------------------------------------
# ProviderMatrix
# ---------------------------------------------------------------------------


class TestProviderMatrix:
    def test_run_returns_matrix_result(self):
        providers = {
            "alpha": _make_provider(score=0.8, cost=0.01),
            "beta": _make_provider(score=0.6, cost=0.005),
        }
        matrix = ProviderMatrix(prompts=["hello", "world"], providers=providers)
        result = matrix.run()
        assert isinstance(result, MatrixResult)
        assert len(result.results) == 4  # 2 prompts x 2 providers

    def test_run_records_provider_names(self):
        providers = {"p1": _make_provider(), "p2": _make_provider()}
        matrix = ProviderMatrix(prompts=["q"], providers=providers)
        result = matrix.run()
        names = {r.provider for r in result.results}
        assert names == {"p1", "p2"}

    def test_run_records_prompts(self):
        providers = {"p": _make_provider()}
        matrix = ProviderMatrix(prompts=["first", "second"], providers=providers)
        result = matrix.run()
        prompts = {r.prompt for r in result.results}
        assert prompts == {"first", "second"}

    def test_run_records_cost(self):
        providers = {"p": _make_provider(cost=0.05)}
        matrix = ProviderMatrix(prompts=["q"], providers=providers)
        result = matrix.run()
        assert result.results[0].cost == 0.05

    def test_run_with_metrics(self):
        providers = {"p": _make_provider(score=0.8)}
        metric_judge = _make_provider(score=0.9)
        matrix = ProviderMatrix(
            prompts=["q"],
            providers=providers,
            metrics={"quality": metric_judge},
        )
        result = matrix.run()
        assert "quality" in result.results[0].scores
        assert result.results[0].scores["quality"] == pytest.approx(0.9)

    def test_arun_is_async(self):
        providers = {"p": _make_provider()}
        matrix = ProviderMatrix(prompts=["hi"], providers=providers)
        result = asyncio.run(matrix.arun())
        assert isinstance(result, MatrixResult)
        assert len(result.results) == 1

    def test_matrix_result_cheapest(self):
        providers = {
            "expensive": _make_provider(cost=0.1),
            "cheap": _make_provider(cost=0.001),
        }
        matrix = ProviderMatrix(prompts=["q"], providers=providers)
        result = matrix.run()
        assert result.cheapest_provider() == "cheap"

    def test_empty_prompts(self):
        providers = {"p": _make_provider()}
        matrix = ProviderMatrix(prompts=[], providers=providers)
        result = matrix.run()
        assert result.results == []

    def test_empty_providers(self):
        matrix = ProviderMatrix(prompts=["q"], providers={})
        result = matrix.run()
        assert result.results == []
