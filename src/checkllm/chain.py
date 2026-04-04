"""Fluent assertion chaining for checkllm.

Usage::

    check.that(output).contains("Python").has_no_pii().max_tokens(200)
    check.that(output).scores_above("relevance", 0.8, query="What is Python?")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from checkllm.check import CheckCollector


class AssertionChain:
    """Fluent wrapper that delegates to CheckCollector methods and returns self."""

    def __init__(self, collector: CheckCollector, output: str) -> None:
        self._collector = collector
        self._output = output

    # --- Deterministic checks ---

    def contains(self, substring: str) -> AssertionChain:
        self._collector.contains(self._output, substring)
        return self

    def not_contains(self, substring: str) -> AssertionChain:
        self._collector.not_contains(self._output, substring)
        return self

    def exact_match(self, expected: str, ignore_case: bool = False) -> AssertionChain:
        self._collector.exact_match(self._output, expected, ignore_case)
        return self

    def starts_with(self, prefix: str) -> AssertionChain:
        self._collector.starts_with(self._output, prefix)
        return self

    def ends_with(self, suffix: str) -> AssertionChain:
        self._collector.ends_with(self._output, suffix)
        return self

    def regex(self, pattern: str) -> AssertionChain:
        self._collector.regex(self._output, pattern)
        return self

    def max_tokens(self, limit: int) -> AssertionChain:
        self._collector.max_tokens(self._output, limit)
        return self

    def min_tokens(self, minimum: int) -> AssertionChain:
        self._collector.min_tokens(self._output, minimum)
        return self

    def word_count(self, min_words: int | None = None, max_words: int | None = None) -> AssertionChain:
        self._collector.word_count(self._output, min_words, max_words)
        return self

    def similarity(self, expected: str, threshold: float = 0.8) -> AssertionChain:
        self._collector.similarity(self._output, expected, threshold)
        return self

    def is_json(self) -> AssertionChain:
        self._collector.is_json(self._output)
        return self

    def is_valid_python(self) -> AssertionChain:
        self._collector.is_valid_python(self._output)
        return self

    def json_schema(self, schema: Type[BaseModel]) -> AssertionChain:
        self._collector.json_schema(self._output, schema)
        return self

    def has_no_pii(self, patterns: list[str] | None = None) -> AssertionChain:
        self._collector.no_pii(self._output, patterns)
        return self

    def language(self, expected: str) -> AssertionChain:
        self._collector.language(self._output, expected)
        return self

    def readability(self, max_grade: float | None = None) -> AssertionChain:
        self._collector.readability(self._output, max_grade)
        return self

    def all_of(self, substrings: list[str]) -> AssertionChain:
        self._collector.all_of(self._output, substrings)
        return self

    def any_of(self, substrings: list[str]) -> AssertionChain:
        self._collector.any_of(self._output, substrings)
        return self

    def none_of(self, substrings: list[str]) -> AssertionChain:
        self._collector.none_of(self._output, substrings)
        return self

    # --- LLM judge checks ---

    def scores_above(self, metric: str, threshold: float, **kwargs: Any) -> AssertionChain:
        """Run a named LLM metric and check it scores above the threshold."""
        method = getattr(self._collector, metric, None)
        if method is None:
            raise AttributeError(
                f"Unknown metric '{metric}'. Use check.that(output).scores_above('metric_name', threshold, ...)"
            )
        method(self._output, threshold=threshold, **kwargs)
        return self
