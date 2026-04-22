from __future__ import annotations

from checkllm.judge import JudgeBackend
from checkllm.metrics.context_relevance import ContextRelevanceMetric
from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.metrics.hallucination import HallucinationMetric
from checkllm.metrics.relevance import RelevanceMetric

from bench.schema import BenchmarkSample, BenchmarkScore, MetricFamily


class CheckllmAdapter:
    """Adapter wrapping CheckLLM metrics for the competitor benchmark harness."""

    framework = "checkllm"

    def __init__(self, judge: JudgeBackend) -> None:
        """Initialise the adapter with a judge backend.

        Args:
            judge: A JudgeBackend (real or mock) used by all metric instances.
        """
        self.judge = judge
        self._hallucination = HallucinationMetric(judge=judge)
        self._faithfulness = FaithfulnessMetric(judge=judge)
        self._context_relevance = ContextRelevanceMetric(judge=judge)
        self._relevance = RelevanceMetric(judge=judge)

    def supports(self, family: MetricFamily) -> bool:
        """Return True if this adapter can score the given metric family.

        Args:
            family: The MetricFamily to test for support.

        Returns:
            True when the adapter has a wired implementation for family.
        """
        return family in {
            MetricFamily.HALLUCINATION,
            MetricFamily.FAITHFULNESS,
            MetricFamily.CONTEXT_RELEVANCE,
            MetricFamily.ANSWER_RELEVANCY,
        }

    async def score(
        self,
        sample: BenchmarkSample,
        family: MetricFamily,
        judge_model: str,
    ) -> BenchmarkScore:
        """Evaluate a sample with the CheckLLM metric for the given family.

        Args:
            sample: The benchmark sample to evaluate.
            family: Which metric family to run.
            judge_model: Identifier of the judge model, stored in the result.

        Returns:
            A BenchmarkScore populated with the metric result.

        Raises:
            NotImplementedError: When family is supported by supports() but has
                no wired implementation yet.
        """
        if family is MetricFamily.HALLUCINATION:
            r = await self._hallucination.evaluate(
                output=sample.answer,
                context=sample.context,
                query=sample.query,
            )
            metric_name = "hallucination"
        elif family is MetricFamily.FAITHFULNESS:
            r = await self._faithfulness.evaluate(
                output=sample.answer, context=sample.context, query=sample.query
            )
            metric_name = "faithfulness"
        elif family is MetricFamily.CONTEXT_RELEVANCE:
            r = await self._context_relevance.evaluate(
                context=sample.context,
                query=sample.query,
                answer=sample.answer or None,
            )
            metric_name = "context_relevance"
        elif family is MetricFamily.ANSWER_RELEVANCY:
            r = await self._relevance.evaluate(output=sample.answer, query=sample.query)
            metric_name = "relevance"
        else:
            raise NotImplementedError(
                f"{family} adapter not yet wired — add support before running this family"
            )

        return BenchmarkScore(
            framework=self.framework,
            dataset=sample.dataset,
            metric_family=family,
            metric_name=metric_name,
            sample_id=sample.sample_id,
            score=r.score,
            passed=r.passed,
            latency_ms=r.latency_ms,
            cost_usd=r.cost,
            judge_model=judge_model,
            reasoning=r.reasoning,
        )
