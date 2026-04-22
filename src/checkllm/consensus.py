"""Consensus judging system — run the same check across multiple LLM judges and aggregate results.

Usage::

    from checkllm.consensus import ConsensusJudge, consensus, AggregationStrategy
    from checkllm.testing import MockJudge

    judges = [("gpt4", gpt4_judge), ("claude", claude_judge)]
    cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
    result = await cj.evaluate("Is this output good?", system_prompt="...")

    # Or use the convenience function:
    result = await consensus(
        output="The capital of France is Paris.",
        metric_name="hallucination",
        judges=judges,
        context="France's capital is Paris.",
    )
"""

from __future__ import annotations

import asyncio
import statistics
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult, JudgeResponse


class AggregationStrategy(str, Enum):
    """Strategy for aggregating multiple judge votes into a single result."""

    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    MEAN = "mean"
    WEIGHTED = "weighted"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class JudgeVote(BaseModel):
    """Individual judge result within a consensus evaluation."""

    judge_name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasoning: str
    cost: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)


class ConsensusResult(BaseModel):
    """Aggregated result from multiple judges."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cost: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    metric_name: str
    strategy: str
    votes: list[JudgeVote]
    agreement_ratio: float = Field(ge=0.0, le=1.0)

    def to_check_result(self) -> CheckResult:
        """Convert to a standard CheckResult for compatibility with the rest of checkllm."""
        return CheckResult(
            passed=self.passed,
            score=self.score,
            reasoning=self.reasoning,
            cost=self.cost,
            latency_ms=self.latency_ms,
            metric_name=self.metric_name,
        )


class ConsensusJudge:
    """Runs the same evaluation across multiple LLM judges and aggregates results.

    Implements the ``JudgeBackend`` protocol so it can be used anywhere a single
    judge is expected.

    Usage::

        judges = [("gpt4", gpt4_judge), ("claude", claude_judge)]
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate("prompt", system_prompt="...")
    """

    def __init__(
        self,
        judges: list[tuple[str, JudgeBackend]],
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN,
        threshold: float = 0.8,
        weights: dict[str, float] | None = None,
    ) -> None:
        if not judges:
            raise ValueError("At least one judge is required")
        self.judges = judges
        self.strategy = AggregationStrategy(strategy)
        self.threshold = threshold
        self.weights = weights or {}
        self.total_cost: float = 0.0

    # ------------------------------------------------------------------
    # JudgeBackend protocol
    # ------------------------------------------------------------------

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Evaluate using consensus and return a JudgeResponse for protocol compatibility."""
        result = await self._evaluate_parallel(prompt, system_prompt)
        return JudgeResponse(
            score=result.score,
            reasoning=result.reasoning,
            cost=result.cost,
        )

    # ------------------------------------------------------------------
    # Core consensus evaluation
    # ------------------------------------------------------------------

    async def evaluate_consensus(
        self,
        prompt: str,
        system_prompt: str | None = None,
        metric_name: str = "consensus",
    ) -> ConsensusResult:
        """Run all judges in parallel and aggregate their results."""
        return await self._evaluate_parallel(prompt, system_prompt, metric_name=metric_name)

    async def _evaluate_parallel(
        self,
        prompt: str,
        system_prompt: str | None = None,
        metric_name: str = "consensus",
    ) -> ConsensusResult:
        """Internal method: fan out to all judges, collect votes, aggregate."""

        async def _call_judge(name: str, judge: JudgeBackend) -> JudgeVote:
            start = time.perf_counter_ns()
            response = await judge.evaluate(prompt, system_prompt)
            elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
            return JudgeVote(
                judge_name=name,
                score=response.score,
                passed=response.score >= self.threshold,
                reasoning=response.reasoning,
                cost=response.cost,
                latency_ms=int(elapsed_ms),
            )

        # Run all judges concurrently
        tasks = [_call_judge(name, judge) for name, judge in self.judges]
        votes: list[JudgeVote] = await asyncio.gather(*tasks)

        # Aggregate
        result = self._aggregate(votes, metric_name)
        self.total_cost += result.cost
        return result

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _aggregate(self, votes: list[JudgeVote], metric_name: str) -> ConsensusResult:
        """Dispatch to the appropriate aggregation strategy."""
        dispatch = {
            AggregationStrategy.MAJORITY: self._agg_majority,
            AggregationStrategy.UNANIMOUS: self._agg_unanimous,
            AggregationStrategy.MEAN: self._agg_mean,
            AggregationStrategy.WEIGHTED: self._agg_weighted,
            AggregationStrategy.MEDIAN: self._agg_median,
            AggregationStrategy.MIN: self._agg_min,
            AggregationStrategy.MAX: self._agg_max,
        }
        passed, score = dispatch[self.strategy](votes)
        agreement_ratio = self._compute_agreement(votes)
        total_cost = sum(v.cost for v in votes)
        max_latency = max(v.latency_ms for v in votes) if votes else 0
        reasoning = self._build_reasoning(votes, passed, score)

        return ConsensusResult(
            passed=passed,
            score=score,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=max_latency,
            metric_name=metric_name,
            strategy=self.strategy.value,
            votes=votes,
            agreement_ratio=agreement_ratio,
        )

    def _agg_majority(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Pass if more than 50% of judges pass."""
        pass_count = sum(1 for v in votes if v.passed)
        passed = pass_count > len(votes) / 2
        score = statistics.mean(v.score for v in votes)
        return passed, round(score, 4)

    def _agg_unanimous(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Pass only if all judges pass."""
        passed = all(v.passed for v in votes)
        score = statistics.mean(v.score for v in votes)
        return passed, round(score, 4)

    def _agg_mean(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Average scores; pass if mean >= threshold."""
        score = statistics.mean(v.score for v in votes)
        passed = score >= self.threshold
        return passed, round(score, 4)

    def _agg_weighted(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Weighted average of scores; pass if weighted mean >= threshold."""
        total_weight = 0.0
        weighted_sum = 0.0
        for v in votes:
            w = self.weights.get(v.judge_name, 1.0)
            weighted_sum += v.score * w
            total_weight += w
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = score >= self.threshold
        return passed, round(score, 4)

    def _agg_median(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Median score; pass if median >= threshold."""
        score = statistics.median(v.score for v in votes)
        passed = score >= self.threshold
        return passed, round(score, 4)

    def _agg_min(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Minimum score (most conservative); pass if min >= threshold."""
        score = min(v.score for v in votes)
        passed = score >= self.threshold
        return passed, round(score, 4)

    def _agg_max(self, votes: list[JudgeVote]) -> tuple[bool, float]:
        """Maximum score (most lenient); pass if max >= threshold."""
        score = max(v.score for v in votes)
        passed = score >= self.threshold
        return passed, round(score, 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_agreement(votes: list[JudgeVote]) -> float:
        """Compute what fraction of judges agree on the majority pass/fail outcome."""
        if not votes:
            return 1.0
        pass_count = sum(1 for v in votes if v.passed)
        fail_count = len(votes) - pass_count
        majority_count = max(pass_count, fail_count)
        return round(majority_count / len(votes), 4)

    @staticmethod
    def _build_reasoning(votes: list[JudgeVote], passed: bool, score: float) -> str:
        """Build a summary reasoning string from all judge votes."""
        parts = [
            f"Consensus ({'PASS' if passed else 'FAIL'}, score={score:.2f}) "
            f"from {len(votes)} judge(s):"
        ]
        for v in votes:
            status = "PASS" if v.passed else "FAIL"
            parts.append(f"  - {v.judge_name}: {status} (score={v.score:.2f}) {v.reasoning}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        names = [n for n, _ in self.judges]
        return (
            f"ConsensusJudge(judges={names!r}, strategy={self.strategy.value!r}, "
            f"threshold={self.threshold}, total_cost=${self.total_cost:.4f})"
        )


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------

# Mapping of metric names to their module paths and class names
_METRIC_MAP: dict[str, tuple[str, str]] = {
    "hallucination": ("checkllm.metrics.hallucination", "HallucinationMetric"),
    "relevance": ("checkllm.metrics.relevance", "RelevanceMetric"),
    "toxicity": ("checkllm.metrics.toxicity", "ToxicityMetric"),
    "fluency": ("checkllm.metrics.fluency", "FluencyMetric"),
    "coherence": ("checkllm.metrics.coherence", "CoherenceMetric"),
    "sentiment": ("checkllm.metrics.sentiment", "SentimentMetric"),
    "correctness": ("checkllm.metrics.correctness", "CorrectnessMetric"),
    "rubric": ("checkllm.metrics.rubric", "RubricMetric"),
}


def _get_metric_class(metric_name: str) -> type:
    """Lazily import and return the metric class for a given name."""
    if metric_name not in _METRIC_MAP:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available: {', '.join(sorted(_METRIC_MAP))}"
        )
    module_path, class_name = _METRIC_MAP[metric_name]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


async def consensus(
    output: str,
    metric_name: str,
    judges: list[tuple[str, JudgeBackend]],
    strategy: str = "mean",
    threshold: float = 0.8,
    weights: dict[str, float] | None = None,
    **metric_kwargs: Any,
) -> ConsensusResult:
    """High-level convenience function for consensus evaluation.

    Creates the specified metric for each judge, runs them in parallel,
    and returns an aggregated ``ConsensusResult``.

    Args:
        output: The LLM output text to evaluate.
        metric_name: Name of a built-in metric (e.g. ``"hallucination"``).
        judges: List of ``(name, judge_backend)`` pairs.
        strategy: Aggregation strategy name (default ``"mean"``).
        threshold: Score threshold for pass/fail (default ``0.8``).
        weights: Optional per-judge weights for the ``"weighted"`` strategy.
        **metric_kwargs: Extra keyword arguments forwarded to the metric's
            ``evaluate()`` method (e.g. ``context=`` for hallucination).

    Returns:
        A ``ConsensusResult`` containing all individual votes and the
        aggregated outcome.

    Usage::

        result = await consensus(
            output="Paris is the capital of France.",
            metric_name="hallucination",
            judges=[("gpt4", gpt4_judge), ("claude", claude_judge)],
            context="France's capital city is Paris.",
        )
    """
    if not judges:
        raise ValueError("At least one judge is required")

    metric_cls = _get_metric_class(metric_name)
    strat = AggregationStrategy(strategy)
    resolved_weights = weights or {}

    async def _run_metric(name: str, judge: JudgeBackend) -> JudgeVote:
        metric_instance = metric_cls(judge=judge, threshold=threshold)
        start = time.perf_counter_ns()
        result: CheckResult = await metric_instance.evaluate(output=output, **metric_kwargs)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        return JudgeVote(
            judge_name=name,
            score=result.score,
            passed=result.passed,
            reasoning=result.reasoning,
            cost=result.cost,
            latency_ms=int(elapsed_ms),
        )

    tasks = [_run_metric(name, judge) for name, judge in judges]
    votes: list[JudgeVote] = await asyncio.gather(*tasks)

    # Re-use the ConsensusJudge aggregation logic
    aggregator = ConsensusJudge(
        judges=judges,
        strategy=strat,
        threshold=threshold,
        weights=resolved_weights,
    )
    return aggregator._aggregate(votes, metric_name=metric_name)
