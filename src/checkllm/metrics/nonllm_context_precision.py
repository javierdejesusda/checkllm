from __future__ import annotations

import re

from checkllm.models import CheckResult


class NonLLMContextPrecisionMetric:
    """Compute context precision without an LLM judge.

    Uses string overlap and similarity between retrieved context chunks
    and a reference answer to determine relevance. Computes Average
    Precision (AP) as the final score.

    This is a zero API cost alternative to LLM-judged context precision.
    """

    def __init__(self, threshold: float = 0.5, similarity_threshold: float = 0.3) -> None:
        """Initialize the metric.

        Args:
            threshold: Minimum AP score to pass.
            similarity_threshold: Minimum word overlap ratio for a context
                chunk to be considered relevant.
        """
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold

    def _word_overlap(self, text_a: str, text_b: str) -> float:
        """Compute word overlap ratio between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Ratio of overlapping words to total unique words.
        """
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / len(words_b)

    async def evaluate(
        self,
        retrieved_contexts: list[str],
        reference: str,
    ) -> CheckResult:
        """Evaluate context precision against a reference.

        For each retrieved context chunk, checks if it has sufficient
        word overlap with the reference to be considered relevant, then
        computes Average Precision.

        Args:
            retrieved_contexts: List of retrieved context strings.
            reference: The reference answer to compare against.

        Returns:
            CheckResult with Average Precision score.
        """
        if not retrieved_contexts or not reference.strip():
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No contexts or empty reference provided",
                cost=0.0,
                latency_ms=0,
                metric_name="nonllm_context_precision",
            )

        relevance: list[bool] = []
        for ctx in retrieved_contexts:
            overlap = self._word_overlap(ctx, reference)
            relevance.append(overlap >= self.similarity_threshold)

        cumulative_relevant = 0
        precision_at_k_sum = 0.0
        for k, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                cumulative_relevant += 1
                precision_at_k_sum += cumulative_relevant / k

        total_relevant = sum(relevance)
        ap = precision_at_k_sum / total_relevant if total_relevant > 0 else 0.0
        ap = max(0.0, min(1.0, ap))
        passed = ap >= self.threshold

        return CheckResult(
            passed=passed,
            score=ap,
            reasoning=(
                f"NonLLM Context Precision (AP): {ap:.4f} "
                f"({total_relevant}/{len(retrieved_contexts)} chunks relevant, "
                f"threshold: {self.threshold})"
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="nonllm_context_precision",
        )
