"""Mixin providing deterministic check methods for CheckCollector."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from checkllm.models import CheckResult

if TYPE_CHECKING:
    from checkllm.deterministic import DeterministicChecks


class DeterministicChecksMixin:
    """Deterministic check methods delegating to self._deterministic."""

    _deterministic: DeterministicChecks
    results: list[CheckResult]

    def contains(self, output: str, substring: str) -> CheckResult:
        result = self._deterministic.contains(output, substring)
        self.results.append(result)
        return result

    def not_contains(self, output: str, substring: str) -> CheckResult:
        result = self._deterministic.not_contains(output, substring)
        self.results.append(result)
        return result

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        result = self._deterministic.max_tokens(output, limit)
        self.results.append(result)
        return result

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        result = self._deterministic.latency(actual_ms, max_ms)
        self.results.append(result)
        return result

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        result = self._deterministic.cost(actual_usd, max_usd)
        self.results.append(result)
        return result

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        result = self._deterministic.json_schema(output, schema)
        self.results.append(result)
        return result

    def regex(self, output: str, pattern: str) -> CheckResult:
        result = self._deterministic.regex(output, pattern)
        self.results.append(result)
        return result

    def exact_match(self, output: str, expected: str, ignore_case: bool = False) -> CheckResult:
        result = self._deterministic.exact_match(output, expected, ignore_case)
        self.results.append(result)
        return result

    def starts_with(self, output: str, prefix: str) -> CheckResult:
        result = self._deterministic.starts_with(output, prefix)
        self.results.append(result)
        return result

    def ends_with(self, output: str, suffix: str) -> CheckResult:
        result = self._deterministic.ends_with(output, suffix)
        self.results.append(result)
        return result

    def min_tokens(self, output: str, minimum: int) -> CheckResult:
        result = self._deterministic.min_tokens(output, minimum)
        self.results.append(result)
        return result

    def word_count(self, output: str, min_words: int | None = None, max_words: int | None = None) -> CheckResult:
        result = self._deterministic.word_count(output, min_words, max_words)
        self.results.append(result)
        return result

    def char_count(self, output: str, min_chars: int | None = None, max_chars: int | None = None) -> CheckResult:
        result = self._deterministic.char_count(output, min_chars, max_chars)
        self.results.append(result)
        return result

    def similarity(self, output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False) -> CheckResult:
        result = self._deterministic.similarity(output, expected, threshold, ignore_case)
        self.results.append(result)
        return result

    def readability(self, output: str, max_grade: float | None = None, min_grade: float | None = None) -> CheckResult:
        result = self._deterministic.readability(output, max_grade, min_grade)
        self.results.append(result)
        return result

    def sentence_count(self, output: str, min_sentences: int | None = None, max_sentences: int | None = None) -> CheckResult:
        result = self._deterministic.sentence_count(output, min_sentences, max_sentences)
        self.results.append(result)
        return result

    def all_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._deterministic.all_of(output, substrings)
        self.results.append(result)
        return result

    def any_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._deterministic.any_of(output, substrings)
        self.results.append(result)
        return result

    def none_of(self, output: str, substrings: list[str]) -> CheckResult:
        result = self._deterministic.none_of(output, substrings)
        self.results.append(result)
        return result

    def is_json(self, output: str) -> CheckResult:
        result = self._deterministic.is_json(output)
        self.results.append(result)
        return result

    def is_valid_python(self, output: str) -> CheckResult:
        result = self._deterministic.is_valid_python(output)
        self.results.append(result)
        return result

    def no_pii(self, output: str, patterns: list[str] | None = None) -> CheckResult:
        result = self._deterministic.no_pii(output, patterns)
        self.results.append(result)
        return result

    def language(self, output: str, expected: str) -> CheckResult:
        result = self._deterministic.language(output, expected)
        self.results.append(result)
        return result

    def greater_than(self, output: str, threshold: float) -> CheckResult:
        result = self._deterministic.greater_than(output, threshold)
        self.results.append(result)
        return result

    def less_than(self, output: str, threshold: float) -> CheckResult:
        result = self._deterministic.less_than(output, threshold)
        self.results.append(result)
        return result

    def between(self, output: str, low: float, high: float) -> CheckResult:
        result = self._deterministic.between(output, low, high)
        self.results.append(result)
        return result

    def bleu(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        result = self._deterministic.bleu(output, reference, threshold)
        self.results.append(result)
        return result

    def rouge_l(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        result = self._deterministic.rouge_l(output, reference, threshold)
        self.results.append(result)
        return result

    def json_field(self, output: str, field_path: str, expected: Any = None, condition: str | None = None) -> CheckResult:
        result = self._deterministic.json_field(output, field_path, expected, condition)
        self.results.append(result)
        return result

    def is_valid_sql(self, output: str) -> CheckResult:
        result = self._deterministic.is_valid_sql(output)
        self.results.append(result)
        return result

    def is_valid_markdown(self, output: str, require_headers: bool = False, require_lists: bool = False, require_code_blocks: bool = False) -> CheckResult:
        result = self._deterministic.is_valid_markdown(output, require_headers, require_lists, require_code_blocks)
        self.results.append(result)
        return result
