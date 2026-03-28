"""Soft assertions — record results without failing the test.

Usage::

    def test_with_soft_checks(check):
        output = my_agent("...")

        # Hard checks — fail the test
        check.contains(output, "Python")

        # Soft checks — recorded but don't fail
        check.expect.relevance(output, query="...")
        check.expect.word_count(output, max_words=100)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from checkllm.models import CheckResult

if TYPE_CHECKING:
    from checkllm.check import CheckCollector

logger = logging.getLogger("checkllm.expect")


class SoftCheckProxy:
    """Proxy that wraps a CheckCollector and marks all results as soft (non-failing).

    Soft check results are stored with ``passed=True`` regardless of the actual
    evaluation outcome.  The original score and reasoning are preserved so they
    still appear in reports and history — they just don't trigger test failure.
    """

    def __init__(self, collector: CheckCollector) -> None:
        self._collector = collector

    def _soften(self, result: CheckResult) -> CheckResult:
        """Mark a result as a soft expectation — always passes but keeps the real score."""
        if not result.passed:
            logger.info(
                "Soft check '%s' would have failed (score=%.2f): %s",
                result.metric_name,
                result.score,
                result.reasoning[:80],
            )
            # Replace the last result in the collector with a softened version
            soft = result.model_copy(
                update={
                    "passed": True,
                    "reasoning": f"[soft] {result.reasoning}",
                }
            )
            # Swap the last result in the collector
            if self._collector.results and self._collector.results[-1] is result:
                self._collector.results[-1] = soft
            return soft
        return result

    # --- Deterministic checks ---

    def contains(self, output: str, substring: str) -> CheckResult:
        result = self._collector.contains(output, substring)
        return self._soften(result)

    def not_contains(self, output: str, substring: str) -> CheckResult:
        result = self._collector.not_contains(output, substring)
        return self._soften(result)

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        result = self._collector.max_tokens(output, limit)
        return self._soften(result)

    def min_tokens(self, output: str, minimum: int) -> CheckResult:
        result = self._collector.min_tokens(output, minimum)
        return self._soften(result)

    def word_count(self, output: str, min_words: int | None = None, max_words: int | None = None) -> CheckResult:
        result = self._collector.word_count(output, min_words, max_words)
        return self._soften(result)

    def char_count(self, output: str, min_chars: int | None = None, max_chars: int | None = None) -> CheckResult:
        result = self._collector.char_count(output, min_chars, max_chars)
        return self._soften(result)

    def regex(self, output: str, pattern: str) -> CheckResult:
        result = self._collector.regex(output, pattern)
        return self._soften(result)

    def exact_match(self, output: str, expected: str, ignore_case: bool = False) -> CheckResult:
        result = self._collector.exact_match(output, expected, ignore_case)
        return self._soften(result)

    def starts_with(self, output: str, prefix: str) -> CheckResult:
        result = self._collector.starts_with(output, prefix)
        return self._soften(result)

    def ends_with(self, output: str, suffix: str) -> CheckResult:
        result = self._collector.ends_with(output, suffix)
        return self._soften(result)

    def similarity(self, output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False) -> CheckResult:
        result = self._collector.similarity(output, expected, threshold, ignore_case)
        return self._soften(result)

    def readability(self, output: str, max_grade: float | None = None, min_grade: float | None = None) -> CheckResult:
        result = self._collector.readability(output, max_grade, min_grade)
        return self._soften(result)

    def sentence_count(self, output: str, min_sentences: int | None = None, max_sentences: int | None = None) -> CheckResult:
        result = self._collector.sentence_count(output, min_sentences, max_sentences)
        return self._soften(result)

    def all_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._collector.all_of(output, substrings)
        return self._soften(result)

    def any_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._collector.any_of(output, substrings)
        return self._soften(result)

    def none_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._collector.none_of(output, substrings)
        return self._soften(result)

    def is_json(self, output: str) -> CheckResult:
        result = self._collector.is_json(output)
        return self._soften(result)

    def is_valid_python(self, output: str) -> CheckResult:
        result = self._collector.is_valid_python(output)
        return self._soften(result)

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        result = self._collector.latency(actual_ms, max_ms)
        return self._soften(result)

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        result = self._collector.cost(actual_usd, max_usd)
        return self._soften(result)

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        result = self._collector.json_schema(output, schema)
        return self._soften(result)

    # --- LLM-as-judge checks ---

    def hallucination(self, output: str, context: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.hallucination(output, context, threshold, runs, system_prompt)
        return self._soften(result)

    def relevance(self, output: str, query: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.relevance(output, query, threshold, runs, system_prompt)
        return self._soften(result)

    def toxicity(self, output: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.toxicity(output, threshold, runs, system_prompt)
        return self._soften(result)

    def rubric(self, output: str, criteria: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.rubric(output, criteria, threshold, runs, system_prompt)
        return self._soften(result)

    def fluency(self, output: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.fluency(output, threshold, runs, system_prompt)
        return self._soften(result)

    def coherence(self, output: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.coherence(output, threshold, runs, system_prompt)
        return self._soften(result)

    def sentiment(self, output: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.sentiment(output, threshold, runs, system_prompt)
        return self._soften(result)

    def correctness(self, output: str, expected: str, threshold: float | None = None, runs: int | None = None, system_prompt: str | None = None) -> CheckResult:
        result = self._collector.correctness(output, expected, threshold, runs, system_prompt)
        return self._soften(result)
