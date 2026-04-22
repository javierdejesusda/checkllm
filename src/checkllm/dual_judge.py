from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

from pydantic import Field

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult, JudgeResponse


class AggregationMethod(Enum):
    """Methods for aggregating scores from two judges."""

    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    REQUIRE_AGREEMENT = "require_agreement"
    WEIGHTED = "weighted"


class DualJudgeResult(JudgeResponse):
    """Extended judge response containing results from both primary and secondary judges.

    Args:
        primary_score: Score from the primary judge.
        secondary_score: Score from the secondary judge.
        primary_reasoning: Reasoning from the primary judge.
        secondary_reasoning: Reasoning from the secondary judge.
        agreement: Whether the two judges agree within the threshold.
        score_difference: Absolute difference between the two scores.
    """

    primary_score: float = Field(ge=0.0, le=1.0)
    secondary_score: float = Field(ge=0.0, le=1.0)
    primary_reasoning: str = ""
    secondary_reasoning: str = ""
    agreement: bool = True
    score_difference: float = Field(ge=0.0)


class DualJudge:
    """Uses two independent judge backends and aggregates their scores.

    Implements the JudgeBackend protocol so it can be used as a drop-in
    replacement anywhere a single judge is expected.

    Args:
        primary: The primary judge backend.
        secondary: The secondary judge backend.
        aggregation: How to combine the two scores.
        agreement_threshold: Maximum allowed score difference for REQUIRE_AGREEMENT.
        primary_weight: Weight for the primary judge when using WEIGHTED aggregation.
    """

    def __init__(
        self,
        primary: JudgeBackend,
        secondary: JudgeBackend,
        aggregation: AggregationMethod = AggregationMethod.AVERAGE,
        agreement_threshold: float = 0.2,
        primary_weight: float = 0.6,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.aggregation = aggregation
        self.agreement_threshold = agreement_threshold
        self.primary_weight = primary_weight

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> DualJudgeResult:
        """Evaluate a prompt using both judges concurrently and aggregate results.

        Args:
            prompt: The evaluation prompt to send to both judges.
            system_prompt: Optional system prompt for both judges.

        Returns:
            DualJudgeResult with aggregated score and both judges' details.
        """
        primary_resp, secondary_resp = await asyncio.gather(
            self.primary.evaluate(prompt, system_prompt),
            self.secondary.evaluate(prompt, system_prompt),
        )

        score_diff = abs(primary_resp.score - secondary_resp.score)
        agreement = score_diff <= self.agreement_threshold

        score = self._aggregate(primary_resp.score, secondary_resp.score, agreement)

        combined_reasoning = (
            f"[Primary judge]: {primary_resp.reasoning}\n"
            f"[Secondary judge]: {secondary_resp.reasoning}"
        )
        if not agreement:
            combined_reasoning += (
                f"\n[Warning]: Judges disagree by {score_diff:.2f} "
                f"(threshold: {self.agreement_threshold:.2f})"
            )

        combined_cost = primary_resp.cost + secondary_resp.cost

        return DualJudgeResult(
            score=score,
            reasoning=combined_reasoning,
            raw_output=None,
            cost=combined_cost,
            primary_score=primary_resp.score,
            secondary_score=secondary_resp.score,
            primary_reasoning=primary_resp.reasoning,
            secondary_reasoning=secondary_resp.reasoning,
            agreement=agreement,
            score_difference=score_diff,
        )

    def _aggregate(self, primary: float, secondary: float, agreement: bool) -> float:
        """Compute the aggregated score from both judges.

        Args:
            primary: Primary judge's score.
            secondary: Secondary judge's score.
            agreement: Whether the judges are within the agreement threshold.

        Returns:
            The aggregated score, clamped to [0.0, 1.0].
        """
        if self.aggregation == AggregationMethod.AVERAGE:
            return (primary + secondary) / 2.0

        if self.aggregation == AggregationMethod.MIN:
            return min(primary, secondary)

        if self.aggregation == AggregationMethod.MAX:
            return max(primary, secondary)

        if self.aggregation == AggregationMethod.WEIGHTED:
            score = self.primary_weight * primary + (1 - self.primary_weight) * secondary
            return max(0.0, min(1.0, score))

        if self.aggregation == AggregationMethod.REQUIRE_AGREEMENT:
            if agreement:
                return (primary + secondary) / 2.0
            return min(primary, secondary)

        return (primary + secondary) / 2.0


class DualJudgeMetric:
    """Wraps any metric class to evaluate with two independent judges in parallel.

    Creates two instances of the given metric class, each with a different
    judge, evaluates in parallel, and aggregates the results.

    Args:
        metric_class: The metric class to instantiate (e.g., HallucinationMetric).
        primary_judge: Judge backend for the first evaluation.
        secondary_judge: Judge backend for the second evaluation.
        aggregation: How to combine the two scores.
        agreement_threshold: Max score difference for agreement.
        primary_weight: Weight for primary judge in WEIGHTED mode.
        **metric_kwargs: Additional keyword arguments passed to the metric constructor.
    """

    def __init__(
        self,
        metric_class: type,
        primary_judge: JudgeBackend,
        secondary_judge: JudgeBackend,
        aggregation: AggregationMethod = AggregationMethod.AVERAGE,
        agreement_threshold: float = 0.2,
        primary_weight: float = 0.6,
        **metric_kwargs: Any,
    ) -> None:
        self.metric_class = metric_class
        self.aggregation = aggregation
        self.agreement_threshold = agreement_threshold
        self.primary_weight = primary_weight
        self._primary_metric = metric_class(judge=primary_judge, **metric_kwargs)
        self._secondary_metric = metric_class(judge=secondary_judge, **metric_kwargs)

    async def evaluate(self, **kwargs: Any) -> CheckResult:
        """Evaluate using both judge-backed metrics concurrently.

        Args:
            **kwargs: Arguments forwarded to the underlying metric's evaluate method.

        Returns:
            Aggregated CheckResult combining both judges' evaluations.
        """
        start = time.perf_counter_ns()

        primary_result, secondary_result = await asyncio.gather(
            self._primary_metric.evaluate(**kwargs),
            self._secondary_metric.evaluate(**kwargs),
        )

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        score_diff = abs(primary_result.score - secondary_result.score)
        agreement = score_diff <= self.agreement_threshold

        score = self._aggregate(primary_result.score, secondary_result.score, agreement)

        combined_reasoning = (
            f"[Primary]: {primary_result.reasoning}\n[Secondary]: {secondary_result.reasoning}"
        )
        if not agreement:
            combined_reasoning += (
                f"\n[Warning]: Judges disagree by {score_diff:.2f} "
                f"(threshold: {self.agreement_threshold:.2f})"
            )

        combined_cost = primary_result.cost + secondary_result.cost
        threshold = primary_result.threshold

        return CheckResult(
            passed=score >= (threshold or 0.0),
            score=score,
            reasoning=combined_reasoning,
            cost=combined_cost,
            latency_ms=int(elapsed_ms),
            metric_name=primary_result.metric_name,
            threshold=threshold,
        )

    def _aggregate(self, primary: float, secondary: float, agreement: bool) -> float:
        """Compute the aggregated score from both metric results.

        Args:
            primary: Primary metric's score.
            secondary: Secondary metric's score.
            agreement: Whether scores are within the agreement threshold.

        Returns:
            The aggregated score.
        """
        if self.aggregation == AggregationMethod.AVERAGE:
            return (primary + secondary) / 2.0

        if self.aggregation == AggregationMethod.MIN:
            return min(primary, secondary)

        if self.aggregation == AggregationMethod.MAX:
            return max(primary, secondary)

        if self.aggregation == AggregationMethod.WEIGHTED:
            score = self.primary_weight * primary + (1 - self.primary_weight) * secondary
            return max(0.0, min(1.0, score))

        if self.aggregation == AggregationMethod.REQUIRE_AGREEMENT:
            if agreement:
                return (primary + secondary) / 2.0
            return min(primary, secondary)

        return (primary + secondary) / 2.0
