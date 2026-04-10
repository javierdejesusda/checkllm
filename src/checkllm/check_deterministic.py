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
        self._fire_before_hook("contains", {"output": output, "substring": substring})
        result = self._deterministic.contains(output, substring)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def not_contains(self, output: str, substring: str) -> CheckResult:
        self._fire_before_hook("not_contains", {"output": output, "substring": substring})
        result = self._deterministic.not_contains(output, substring)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        self._fire_before_hook("max_tokens", {"output": output, "limit": limit})
        result = self._deterministic.max_tokens(output, limit)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        self._fire_before_hook("latency", {"actual_ms": actual_ms, "max_ms": max_ms})
        result = self._deterministic.latency(actual_ms, max_ms)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        self._fire_before_hook("cost", {"actual_usd": actual_usd, "max_usd": max_usd})
        result = self._deterministic.cost(actual_usd, max_usd)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        self._fire_before_hook("json_schema", {"output": output, "schema": schema})
        result = self._deterministic.json_schema(output, schema)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def regex(self, output: str, pattern: str) -> CheckResult:
        self._fire_before_hook("regex", {"output": output, "pattern": pattern})
        result = self._deterministic.regex(output, pattern)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def exact_match(self, output: str, expected: str, ignore_case: bool = False) -> CheckResult:
        self._fire_before_hook("exact_match", {"output": output, "expected": expected, "ignore_case": ignore_case})
        result = self._deterministic.exact_match(output, expected, ignore_case)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def starts_with(self, output: str, prefix: str) -> CheckResult:
        self._fire_before_hook("starts_with", {"output": output, "prefix": prefix})
        result = self._deterministic.starts_with(output, prefix)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def ends_with(self, output: str, suffix: str) -> CheckResult:
        self._fire_before_hook("ends_with", {"output": output, "suffix": suffix})
        result = self._deterministic.ends_with(output, suffix)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def min_tokens(self, output: str, minimum: int) -> CheckResult:
        self._fire_before_hook("min_tokens", {"output": output, "minimum": minimum})
        result = self._deterministic.min_tokens(output, minimum)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def word_count(self, output: str, min_words: int | None = None, max_words: int | None = None) -> CheckResult:
        self._fire_before_hook("word_count", {"output": output, "min_words": min_words, "max_words": max_words})
        result = self._deterministic.word_count(output, min_words, max_words)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def char_count(self, output: str, min_chars: int | None = None, max_chars: int | None = None) -> CheckResult:
        self._fire_before_hook("char_count", {"output": output, "min_chars": min_chars, "max_chars": max_chars})
        result = self._deterministic.char_count(output, min_chars, max_chars)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def similarity(self, output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False) -> CheckResult:
        self._fire_before_hook("similarity", {"output": output, "expected": expected, "threshold": threshold, "ignore_case": ignore_case})
        result = self._deterministic.similarity(output, expected, threshold, ignore_case)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def readability(self, output: str, max_grade: float | None = None, min_grade: float | None = None) -> CheckResult:
        self._fire_before_hook("readability", {"output": output, "max_grade": max_grade, "min_grade": min_grade})
        result = self._deterministic.readability(output, max_grade, min_grade)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def sentence_count(self, output: str, min_sentences: int | None = None, max_sentences: int | None = None) -> CheckResult:
        self._fire_before_hook("sentence_count", {"output": output, "min_sentences": min_sentences, "max_sentences": max_sentences})
        result = self._deterministic.sentence_count(output, min_sentences, max_sentences)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def all_of(self, output: str, substrings: list[str]) -> CheckResult:
        self._fire_before_hook("all_of", {"output": output, "substrings": substrings})
        result = self._deterministic.all_of(output, substrings)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def any_of(self, output: str, substrings: list[str]) -> CheckResult:
        self._fire_before_hook("any_of", {"output": output, "substrings": substrings})
        result = self._deterministic.any_of(output, substrings)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def none_of(self, output: str, substrings: list[str]) -> CheckResult:
        self._fire_before_hook("none_of", {"output": output, "substrings": substrings})
        result = self._deterministic.none_of(output, substrings)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_json(self, output: str) -> CheckResult:
        self._fire_before_hook("is_json", {"output": output})
        result = self._deterministic.is_json(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_valid_python(self, output: str) -> CheckResult:
        self._fire_before_hook("is_valid_python", {"output": output})
        result = self._deterministic.is_valid_python(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def no_pii(self, output: str, patterns: list[str] | None = None) -> CheckResult:
        self._fire_before_hook("no_pii", {"output": output, "patterns": patterns})
        result = self._deterministic.no_pii(output, patterns)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def language(self, output: str, expected: str) -> CheckResult:
        self._fire_before_hook("language", {"output": output, "expected": expected})
        result = self._deterministic.language(output, expected)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def greater_than(self, output: str, threshold: float) -> CheckResult:
        self._fire_before_hook("greater_than", {"output": output, "threshold": threshold})
        result = self._deterministic.greater_than(output, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def less_than(self, output: str, threshold: float) -> CheckResult:
        self._fire_before_hook("less_than", {"output": output, "threshold": threshold})
        result = self._deterministic.less_than(output, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def between(self, output: str, low: float, high: float) -> CheckResult:
        self._fire_before_hook("between", {"output": output, "low": low, "high": high})
        result = self._deterministic.between(output, low, high)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def bleu(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        self._fire_before_hook("bleu", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.bleu(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def rouge_l(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        self._fire_before_hook("rouge_l", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.rouge_l(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def json_field(self, output: str, field_path: str, expected: Any = None, condition: str | None = None) -> CheckResult:
        self._fire_before_hook("json_field", {"output": output, "field_path": field_path, "expected": expected, "condition": condition})
        result = self._deterministic.json_field(output, field_path, expected, condition)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_valid_sql(self, output: str) -> CheckResult:
        self._fire_before_hook("is_valid_sql", {"output": output})
        result = self._deterministic.is_valid_sql(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_valid_markdown(self, output: str, require_headers: bool = False, require_lists: bool = False, require_code_blocks: bool = False) -> CheckResult:
        self._fire_before_hook("is_valid_markdown", {"output": output, "require_headers": require_headers, "require_lists": require_lists, "require_code_blocks": require_code_blocks})
        result = self._deterministic.is_valid_markdown(output, require_headers, require_lists, require_code_blocks)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def icontains(self, output: str, substring: str) -> CheckResult:
        self._fire_before_hook("icontains", {"output": output, "substring": substring})
        result = self._deterministic.icontains(output, substring)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def icontains_any(self, output: str, substrings: list[str]) -> CheckResult:
        self._fire_before_hook("icontains_any", {"output": output, "substrings": substrings})
        result = self._deterministic.icontains_any(output, substrings)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def icontains_all(self, output: str, substrings: list[str]) -> CheckResult:
        self._fire_before_hook("icontains_all", {"output": output, "substrings": substrings})
        result = self._deterministic.icontains_all(output, substrings)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_html(self, output: str) -> CheckResult:
        self._fire_before_hook("is_html", {"output": output})
        result = self._deterministic.is_html(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def contains_html(self, output: str) -> CheckResult:
        self._fire_before_hook("contains_html", {"output": output})
        result = self._deterministic.contains_html(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_xml(self, output: str) -> CheckResult:
        self._fire_before_hook("is_xml", {"output": output})
        result = self._deterministic.is_xml(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def contains_xml(self, output: str) -> CheckResult:
        self._fire_before_hook("contains_xml", {"output": output})
        result = self._deterministic.contains_xml(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_refusal(self, output: str) -> CheckResult:
        self._fire_before_hook("is_refusal", {"output": output})
        result = self._deterministic.is_refusal(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def levenshtein(self, output: str, reference: str, threshold: float = 0.7) -> CheckResult:
        self._fire_before_hook("levenshtein", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.levenshtein(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def meteor(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        self._fire_before_hook("meteor", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.meteor(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def perplexity_check(self, output: str, max_perplexity: float = 50.0) -> CheckResult:
        self._fire_before_hook("perplexity_check", {"output": output, "max_perplexity": max_perplexity})
        result = self._deterministic.perplexity_check(output, max_perplexity)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_valid_yaml(self, output: str) -> CheckResult:
        self._fire_before_hook("is_valid_yaml", {"output": output})
        result = self._deterministic.is_valid_yaml(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def has_citations(self, output: str, min_count: int = 1) -> CheckResult:
        self._fire_before_hook("has_citations", {"output": output, "min_count": min_count})
        result = self._deterministic.has_citations(output, min_count)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def no_repetition(self, output: str, max_ngram_repeat: int = 3) -> CheckResult:
        self._fire_before_hook("no_repetition", {"output": output, "max_ngram_repeat": max_ngram_repeat})
        result = self._deterministic.no_repetition(output, max_ngram_repeat)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def semantic_similarity(self, output: str, reference: str, threshold: float = 0.7) -> CheckResult:
        self._fire_before_hook("semantic_similarity", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.semantic_similarity(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def is_valid_url(self, output: str) -> CheckResult:
        self._fire_before_hook("is_valid_url", {"output": output})
        result = self._deterministic.is_valid_url(output)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def has_structure(self, output: str, elements: list[str]) -> CheckResult:
        self._fire_before_hook("has_structure", {"output": output, "elements": elements})
        result = self._deterministic.has_structure(output, elements)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def gleu(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        self._fire_before_hook("gleu", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.gleu(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def chrf(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        self._fire_before_hook("chrf", {"output": output, "reference": reference, "threshold": threshold})
        result = self._deterministic.chrf(output, reference, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def latency_check(self, start_time: float, end_time: float, max_ms: float = 5000.0) -> CheckResult:
        self._fire_before_hook("latency_check", {"start_time": start_time, "end_time": end_time, "max_ms": max_ms})
        result = self._deterministic.latency_check(start_time, end_time, max_ms)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def cost_check(self, input_tokens: int, output_tokens: int, model: str, max_cost: float = 1.0) -> CheckResult:
        self._fire_before_hook("cost_check", {"input_tokens": input_tokens, "output_tokens": output_tokens, "model": model, "max_cost": max_cost})
        result = self._deterministic.cost_check(input_tokens, output_tokens, model, max_cost)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def string_distance(self, output: str, reference: str, method: str = "levenshtein", threshold: float = 0.7) -> CheckResult:
        self._fire_before_hook("string_distance", {"output": output, "reference": reference, "method": method, "threshold": threshold})
        result = self._deterministic.string_distance(output, reference, method, threshold)
        self.results.append(result)
        self._fire_after_hook(result)
        return result

    def exact_match_strict(self, output: str, reference: str, ignore_case: bool = False, ignore_whitespace: bool = False) -> CheckResult:
        self._fire_before_hook("exact_match_strict", {"output": output, "reference": reference, "ignore_case": ignore_case, "ignore_whitespace": ignore_whitespace})
        result = self._deterministic.exact_match_strict(output, reference, ignore_case, ignore_whitespace)
        self.results.append(result)
        self._fire_after_hook(result)
        return result
