from __future__ import annotations

import time

from bench.schema import BenchmarkSample, BenchmarkScore, MetricFamily


class RagasAdapter:
    """Adapter that wraps Ragas evaluation metrics behind the FrameworkAdapter protocol."""

    framework = "ragas"

    def __init__(self, llm) -> None:
        """`llm` is a LangChain-compatible LLM (BaseChatModel or test stub)."""
        self._llm = llm

    def supports(self, family: MetricFamily) -> bool:
        """Return True if this adapter can evaluate the given metric family.

        Args:
            family: The metric family to check.

        Returns:
            True when Ragas supports the metric family.
        """
        return family in {
            MetricFamily.FAITHFULNESS,
            MetricFamily.ANSWER_RELEVANCY,
            MetricFamily.CONTEXT_RELEVANCE,
            MetricFamily.HALLUCINATION,
        }

    async def score(
        self,
        sample: BenchmarkSample,
        family: MetricFamily,
        judge_model: str,
    ) -> BenchmarkScore:
        """Score a single sample using the specified Ragas metric.

        Args:
            sample: The benchmark sample to evaluate.
            family: Which metric family to evaluate.
            judge_model: Identifier for the judge model being used.

        Returns:
            A BenchmarkScore with the evaluation result.

        Raises:
            NotImplementedError: If the metric family is not supported.
        """
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        wrapped = LangchainLLMWrapper(self._llm)

        ragas_sample = SingleTurnSample(
            user_input=sample.query,
            response=sample.answer,
            retrieved_contexts=[sample.context] if sample.context else [],
            reference=sample.context if sample.context else None,
        )

        metric_map = {
            MetricFamily.FAITHFULNESS: (faithfulness, "faithfulness"),
            MetricFamily.HALLUCINATION: (faithfulness, "faithfulness_as_hallucination"),
            MetricFamily.ANSWER_RELEVANCY: (answer_relevancy, "answer_relevancy"),
            MetricFamily.CONTEXT_RELEVANCE: (context_precision, "context_precision"),
        }
        if family not in metric_map:
            raise NotImplementedError(f"{family} not supported by ragas adapter")

        metric_obj, metric_name = metric_map[family]
        metric_obj.llm = wrapped

        start = time.perf_counter_ns()
        try:
            score_val = float(await metric_obj.single_turn_ascore(ragas_sample))
        except Exception as exc:
            return BenchmarkScore(
                framework=self.framework,
                dataset=sample.dataset,
                metric_family=family,
                metric_name=metric_name,
                sample_id=sample.sample_id,
                score=0.0,
                passed=False,
                latency_ms=(time.perf_counter_ns() - start) // 1_000_000,
                cost_usd=0.0,
                judge_model=judge_model,
                reasoning=f"error: {exc}",
            )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        score_val = max(0.0, min(1.0, score_val))

        return BenchmarkScore(
            framework=self.framework,
            dataset=sample.dataset,
            metric_family=family,
            metric_name=metric_name,
            sample_id=sample.sample_id,
            score=score_val,
            passed=score_val >= 0.5,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
            judge_model=judge_model,
            reasoning="",
        )
