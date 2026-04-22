from __future__ import annotations

import re

from checkllm.models import CheckResult


class FaithfulnessHHEMMetric:
    """NLI-based faithfulness metric inspired by Ragas FaithfulnesswithHHEM.

    Extracts claims from the response, then checks each claim against
    the context using textual entailment heuristics (lexical overlap and
    keyword matching) as an NLI proxy. This is a zero API cost
    alternative to LLM-judged faithfulness.

    Score = entailed_claims / total_claims
    """

    def __init__(self, threshold: float = 0.7, entailment_threshold: float = 0.4) -> None:
        """Initialize the metric.

        Args:
            threshold: Minimum faithfulness score to pass.
            entailment_threshold: Minimum keyword overlap ratio for a
                claim to be considered entailed by context.
        """
        self.threshold = threshold
        self.entailment_threshold = entailment_threshold

    def _extract_claims(self, text: str) -> list[str]:
        """Extract individual claims (sentences) from the response.

        Args:
            text: The response text.

        Returns:
            List of claim strings.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims if claims else [text.strip()] if text.strip() else []

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text, filtering stop words.

        Args:
            text: Input text.

        Returns:
            Set of lowercase keyword strings.
        """
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "than",
            "too",
            "very",
            "just",
            "about",
            "also",
            "then",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
        }
        words = set(re.findall(r"\w+", text.lower()))
        return words - stop_words

    def _check_entailment(self, claim: str, context: str) -> float:
        """Check if a claim is entailed by the context using keyword overlap.

        Args:
            claim: A single claim to verify.
            context: The source context.

        Returns:
            Entailment score from 0.0 to 1.0.
        """
        claim_keywords = self._extract_keywords(claim)
        context_keywords = self._extract_keywords(context)

        if not claim_keywords:
            return 1.0

        overlap = claim_keywords & context_keywords
        keyword_ratio = len(overlap) / len(claim_keywords)

        claim_lower = claim.lower()
        context_lower = context.lower()
        claim_words = claim_lower.split()
        bigrams = [f"{claim_words[i]} {claim_words[i + 1]}" for i in range(len(claim_words) - 1)]
        bigram_matches = sum(1 for bg in bigrams if bg in context_lower)
        bigram_ratio = bigram_matches / len(bigrams) if bigrams else 0.0

        return 0.7 * keyword_ratio + 0.3 * bigram_ratio

    async def evaluate(
        self,
        output: str,
        context: str,
    ) -> CheckResult:
        """Evaluate faithfulness of output against context using NLI heuristics.

        Args:
            output: The model response to evaluate.
            context: The source context that the response should be
                faithful to.

        Returns:
            CheckResult with faithfulness score.
        """
        if not output.strip() or not context.strip():
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="Empty output or context",
                cost=0.0,
                latency_ms=0,
                metric_name="faithfulness_hhem",
            )

        claims = self._extract_claims(output)
        if not claims:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No claims extracted from output",
                cost=0.0,
                latency_ms=0,
                metric_name="faithfulness_hhem",
            )

        entailed_count = 0
        for claim in claims:
            entailment_score = self._check_entailment(claim, context)
            if entailment_score >= self.entailment_threshold:
                entailed_count += 1

        score = entailed_count / len(claims)
        score = max(0.0, min(1.0, score))
        passed = score >= self.threshold

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"FaithfulnessHHEM: {score:.4f} "
                f"({entailed_count}/{len(claims)} claims entailed, "
                f"threshold: {self.threshold})"
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="faithfulness_hhem",
        )
