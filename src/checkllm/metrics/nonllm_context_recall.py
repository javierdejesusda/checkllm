from __future__ import annotations

import re

from checkllm.models import CheckResult


class NonLLMContextRecallMetric:
    """Compute context recall without an LLM judge.

    For each sentence in the reference, checks whether any retrieved
    context chunk has sufficient string similarity. Computes recall as
    the fraction of reference sentences that are covered by the context.

    This is a zero API cost alternative to LLM-judged context recall.
    """

    def __init__(self, threshold: float = 0.5, similarity_threshold: float = 0.3) -> None:
        """Initialize the metric.

        Args:
            threshold: Minimum recall score to pass.
            similarity_threshold: Minimum word overlap ratio for a
                reference sentence to be considered covered.
        """
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold

    def _sentence_split(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of non-empty sentence strings.
        """
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _word_overlap(self, text_a: str, text_b: str) -> float:
        """Compute word overlap ratio between two texts.

        Args:
            text_a: First text (context chunk).
            text_b: Second text (reference sentence).

        Returns:
            Ratio of overlapping words relative to the reference sentence words.
        """
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / len(words_b)

    async def evaluate(
        self,
        retrieved_contexts: list[str],
        reference: str,
    ) -> CheckResult:
        """Evaluate context recall against a reference.

        For each reference sentence, checks if any retrieved context
        chunk has sufficient word overlap to cover it.

        Args:
            retrieved_contexts: List of retrieved context strings.
            reference: The reference answer whose sentences should
                be covered by context.

        Returns:
            CheckResult with recall score.
        """
        ref_sentences = self._sentence_split(reference)

        if not ref_sentences:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No sentences found in reference",
                cost=0.0,
                latency_ms=0,
                metric_name="nonllm_context_recall",
            )

        if not retrieved_contexts:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No retrieved contexts provided",
                cost=0.0,
                latency_ms=0,
                metric_name="nonllm_context_recall",
            )

        combined_context = " ".join(retrieved_contexts)
        matched = 0
        for sentence in ref_sentences:
            for ctx in retrieved_contexts:
                overlap = self._word_overlap(ctx, sentence)
                if overlap >= self.similarity_threshold:
                    matched += 1
                    break

        recall = matched / len(ref_sentences)
        recall = max(0.0, min(1.0, recall))
        passed = recall >= self.threshold

        return CheckResult(
            passed=passed,
            score=recall,
            reasoning=(
                f"NonLLM Context Recall: {recall:.4f} "
                f"({matched}/{len(ref_sentences)} reference sentences covered, "
                f"threshold: {self.threshold})"
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="nonllm_context_recall",
        )
