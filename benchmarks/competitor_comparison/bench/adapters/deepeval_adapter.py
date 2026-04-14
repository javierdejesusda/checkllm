from __future__ import annotations

import time

from bench.schema import BenchmarkSample, BenchmarkScore, MetricFamily


class DeepEvalAdapter:
    framework = "deepeval"

    def __init__(self, model) -> None:
        """`model` is a deepeval.models.base_model.DeepEvalBaseLLM instance."""
        self._model = model

    def supports(self, family: MetricFamily) -> bool:
        return family in {
            MetricFamily.HALLUCINATION,
            MetricFamily.FAITHFULNESS,
            MetricFamily.ANSWER_RELEVANCY,
            MetricFamily.CONTEXT_RELEVANCE,
        }

    async def score(
        self,
        sample: BenchmarkSample,
        family: MetricFamily,
        judge_model: str,
    ) -> BenchmarkScore:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=sample.query,
            actual_output=sample.answer,
            context=[sample.context] if sample.context else None,
            retrieval_context=[sample.context] if sample.context else None,
            expected_output=None,
        )

        metric_map = {
            MetricFamily.HALLUCINATION: (HallucinationMetric, "hallucination"),
            MetricFamily.FAITHFULNESS: (FaithfulnessMetric, "faithfulness"),
            MetricFamily.ANSWER_RELEVANCY: (AnswerRelevancyMetric, "answer_relevancy"),
            MetricFamily.CONTEXT_RELEVANCE: (ContextualRelevancyMetric, "contextual_relevancy"),
        }
        if family not in metric_map:
            raise NotImplementedError(f"{family} not supported by deepeval adapter")

        cls, name = metric_map[family]
        start = time.perf_counter_ns()
        try:
            metric = cls(model=self._model, threshold=0.5, async_mode=False, strict_mode=False)
            metric.measure(test_case)
        except Exception as exc:
            return BenchmarkScore(
                framework=self.framework,
                dataset=sample.dataset,
                metric_family=family,
                metric_name=name,
                sample_id=sample.sample_id,
                score=0.0,
                passed=False,
                latency_ms=(time.perf_counter_ns() - start) // 1_000_000,
                cost_usd=0.0,
                judge_model=judge_model,
                reasoning=f"error: {exc}",
            )

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        raw = float(getattr(metric, "score", 0.0) or 0.0)
        raw = max(0.0, min(1.0, raw))
        # DeepEval hallucination: higher = more hallucination; invert to match "1=faithful"
        score_val = 1.0 - raw if family is MetricFamily.HALLUCINATION else raw
        reason = str(getattr(metric, "reason", ""))

        return BenchmarkScore(
            framework=self.framework,
            dataset=sample.dataset,
            metric_family=family,
            metric_name=name,
            sample_id=sample.sample_id,
            score=score_val,
            passed=score_val >= 0.5,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
            judge_model=judge_model,
            reasoning=reason,
        )
