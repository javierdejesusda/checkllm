from __future__ import annotations

import re

from checkllm.models import CheckResult


class QuotedSpansAlignmentMetric:
    """Check if quoted spans in the response appear verbatim in source contexts.

    Extracts all quoted text (between quotation marks) from the response
    and checks each quote against the retrieved contexts for verbatim match.

    Score = matched_quotes / total_quotes
    """

    def __init__(self, threshold: float = 0.8, min_quote_length: int = 3) -> None:
        """Initialize the metric.

        Args:
            threshold: Minimum score to pass.
            min_quote_length: Minimum character length for a quoted span
                to be considered (ignores very short quotes).
        """
        self.threshold = threshold
        self.min_quote_length = min_quote_length

    def _extract_quotes(self, text: str) -> list[str]:
        """Extract quoted text from the response.

        Supports double quotes, single quotes, and smart/curly quotes.

        Args:
            text: The response text to extract quotes from.

        Returns:
            List of quoted text strings.
        """
        patterns = [
            r'"([^"]+)"',
            r"\u201c([^\u201d]+)\u201d",
            r"\u2018([^\u2019]+)\u2019",
        ]
        quotes: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            quotes.extend(m.strip() for m in matches if len(m.strip()) >= self.min_quote_length)
        return quotes

    async def evaluate(
        self,
        output: str,
        retrieved_contexts: list[str],
    ) -> CheckResult:
        """Evaluate whether quoted spans in output match source contexts.

        Args:
            output: The model response containing quoted spans.
            retrieved_contexts: List of source context strings to
                verify quotes against.

        Returns:
            CheckResult with quote alignment score.
        """
        if not output.strip():
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="Empty output, no quotes to check",
                cost=0.0,
                latency_ms=0,
                metric_name="quoted_spans_alignment",
            )

        quotes = self._extract_quotes(output)
        if not quotes:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No quoted spans found in output",
                cost=0.0,
                latency_ms=0,
                metric_name="quoted_spans_alignment",
            )

        if not retrieved_contexts:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Found {len(quotes)} quote(s) but no contexts to verify against",
                cost=0.0,
                latency_ms=0,
                metric_name="quoted_spans_alignment",
            )

        combined_context = " ".join(retrieved_contexts)
        matched = 0
        unmatched_quotes: list[str] = []

        for quote in quotes:
            if quote in combined_context:
                matched += 1
            else:
                quote_lower = quote.lower()
                context_lower = combined_context.lower()
                if quote_lower in context_lower:
                    matched += 1
                else:
                    unmatched_quotes.append(quote[:50])

        score = matched / len(quotes)
        score = max(0.0, min(1.0, score))
        passed = score >= self.threshold

        detail = ""
        if unmatched_quotes:
            preview = "; ".join(f'"{q}"' for q in unmatched_quotes[:3])
            detail = f" Unmatched: {preview}"

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"QuotedSpansAlignment: {score:.4f} "
                f"({matched}/{len(quotes)} quotes verified, "
                f"threshold: {self.threshold}).{detail}"
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="quoted_spans_alignment",
        )
