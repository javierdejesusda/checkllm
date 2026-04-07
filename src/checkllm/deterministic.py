from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Type
from urllib.parse import urlparse

import tiktoken
from pydantic import BaseModel, ValidationError

from checkllm.models import CheckResult


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio (0.0 = completely different, 1.0 = identical)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    # Wagner-Fischer algorithm
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    distance = prev[len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def _flesch_kincaid_grade(text: str) -> float:
    """Compute approximate Flesch-Kincaid Grade Level."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    syllable_count = sum(_count_syllables(w) for w in words)
    num_sentences = len(sentences)
    num_words = len(words)
    return (
        0.39 * (num_words / num_sentences)
        + 11.8 * (syllable_count / num_words)
        - 15.59
    )


def _count_syllables(word: str) -> int:
    """Rough syllable count for English words."""
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


class DeterministicChecks:
    """Deterministic checks that run locally with zero API calls."""

    def contains(self, output: str, substring: str) -> CheckResult:
        found = substring in output
        return CheckResult(
            passed=found,
            score=1.0 if found else 0.0,
            reasoning=f"Substring '{substring}' {'found' if found else 'not found'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="contains",
            input_preview=output[:200],
        )

    def not_contains(self, output: str, substring: str) -> CheckResult:
        absent = substring not in output
        return CheckResult(
            passed=absent,
            score=1.0 if absent else 0.0,
            reasoning=f"Substring '{substring}' {'not found' if absent else 'found'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="not_contains",
            input_preview=output[:200],
        )

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(output))
        passed = token_count <= limit
        return CheckResult(
            passed=passed,
            score=min(1.0, limit / max(token_count, 1)),
            reasoning=f"Token count: {token_count}, limit: {limit}",
            cost=0.0,
            latency_ms=0,
            metric_name="max_tokens",
            input_preview=output[:200],
        )

    def min_tokens(self, output: str, minimum: int) -> CheckResult:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(output))
        passed = token_count >= minimum
        return CheckResult(
            passed=passed,
            score=min(1.0, token_count / max(minimum, 1)),
            reasoning=f"Token count: {token_count}, minimum: {minimum}",
            cost=0.0,
            latency_ms=0,
            metric_name="min_tokens",
            input_preview=output[:200],
        )

    def word_count(self, output: str, min_words: int | None = None, max_words: int | None = None) -> CheckResult:
        count = len(output.split())
        passed = True
        reasons = [f"Word count: {count}"]
        if min_words is not None:
            if count < min_words:
                passed = False
            reasons.append(f"min: {min_words}")
        if max_words is not None:
            if count > max_words:
                passed = False
            reasons.append(f"max: {max_words}")

        if min_words is not None and max_words is not None:
            target = (min_words + max_words) / 2
            score = max(0.0, 1.0 - abs(count - target) / max(target, 1))
        elif max_words is not None:
            score = min(1.0, max_words / max(count, 1))
        elif min_words is not None:
            score = min(1.0, count / max(min_words, 1))
        else:
            score = 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=", ".join(reasons),
            cost=0.0,
            latency_ms=0,
            metric_name="word_count",
            input_preview=output[:200],
        )

    def char_count(self, output: str, min_chars: int | None = None, max_chars: int | None = None) -> CheckResult:
        count = len(output)
        passed = True
        reasons = [f"Char count: {count}"]
        if min_chars is not None:
            if count < min_chars:
                passed = False
            reasons.append(f"min: {min_chars}")
        if max_chars is not None:
            if count > max_chars:
                passed = False
            reasons.append(f"max: {max_chars}")

        if min_chars is not None and max_chars is not None:
            target = (min_chars + max_chars) / 2
            score = max(0.0, 1.0 - abs(count - target) / max(target, 1))
        elif max_chars is not None:
            score = min(1.0, max_chars / max(count, 1))
        elif min_chars is not None:
            score = min(1.0, count / max(min_chars, 1))
        else:
            score = 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=", ".join(reasons),
            cost=0.0,
            latency_ms=0,
            metric_name="char_count",
            input_preview=output[:200],
        )

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        passed = actual_ms <= max_ms
        return CheckResult(
            passed=passed,
            score=min(1.0, max_ms / max(actual_ms, 1)),
            reasoning=f"Latency: {actual_ms}ms, limit: {max_ms}ms",
            cost=0.0,
            latency_ms=0,
            metric_name="latency",
        )

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        passed = actual_usd <= max_usd
        return CheckResult(
            passed=passed,
            score=min(1.0, max_usd / max(actual_usd, 0.0001)),
            reasoning=f"Cost: ${actual_usd:.4f}, limit: ${max_usd:.4f}",
            cost=0.0,
            latency_ms=0,
            metric_name="cost",
        )

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        try:
            parsed = json.loads(output)
            schema.model_validate(parsed)
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning=f"Output is valid {schema.__name__} JSON",
                cost=0.0,
                latency_ms=0,
                metric_name="json_schema",
                input_preview=output[:200],
            )
        except (json.JSONDecodeError, ValidationError) as e:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"JSON schema validation failed: {e}",
                cost=0.0,
                latency_ms=0,
                metric_name="json_schema",
                input_preview=output[:200],
            )

    def regex(self, output: str, pattern: str) -> CheckResult:
        match = re.search(pattern, output)
        passed = match is not None
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Pattern '{pattern}' {'matched' if passed else 'not matched'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="regex",
            input_preview=output[:200],
        )

    def exact_match(self, output: str, expected: str, ignore_case: bool = False) -> CheckResult:
        a, b = (output.lower(), expected.lower()) if ignore_case else (output, expected)
        passed = a.strip() == b.strip()
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Output {'matches' if passed else 'does not match'} expected",
            cost=0.0,
            latency_ms=0,
            metric_name="exact_match",
            input_preview=output[:200],
        )

    def starts_with(self, output: str, prefix: str) -> CheckResult:
        passed = output.startswith(prefix)
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Output {'starts' if passed else 'does not start'} with '{prefix}'",
            cost=0.0,
            latency_ms=0,
            metric_name="starts_with",
            input_preview=output[:200],
        )

    def ends_with(self, output: str, suffix: str) -> CheckResult:
        passed = output.endswith(suffix)
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Output {'ends' if passed else 'does not end'} with '{suffix}'",
            cost=0.0,
            latency_ms=0,
            metric_name="ends_with",
            input_preview=output[:200],
        )

    def similarity(self, output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False) -> CheckResult:
        """Lexical similarity using Levenshtein ratio."""
        a, b = (output.lower(), expected.lower()) if ignore_case else (output, expected)
        ratio = _levenshtein_ratio(a, b)
        passed = ratio >= threshold
        return CheckResult(
            passed=passed,
            score=ratio,
            reasoning=f"Levenshtein similarity: {ratio:.3f} (threshold: {threshold})",
            cost=0.0,
            latency_ms=0,
            metric_name="similarity",
            input_preview=output[:200],
            threshold=threshold,
        )

    def readability(self, output: str, max_grade: float | None = None, min_grade: float | None = None) -> CheckResult:
        """Flesch-Kincaid Grade Level check."""
        grade = _flesch_kincaid_grade(output)
        passed = True
        reasons = [f"Flesch-Kincaid grade: {grade:.1f}"]
        if max_grade is not None:
            if grade > max_grade:
                passed = False
            reasons.append(f"max: {max_grade}")
        if min_grade is not None:
            if grade < min_grade:
                passed = False
            reasons.append(f"min: {min_grade}")

        # Score: 1.0 if within range, lower as it deviates
        if max_grade is not None and min_grade is not None:
            if min_grade <= grade <= max_grade:
                score = 1.0
            else:
                dist = min(abs(grade - min_grade), abs(grade - max_grade))
                score = max(0.0, 1.0 - dist / 10.0)
        elif max_grade is not None:
            score = min(1.0, max_grade / max(grade, 0.1)) if grade > max_grade else 1.0
        elif min_grade is not None:
            score = min(1.0, grade / max(min_grade, 0.1)) if grade < min_grade else 1.0
        else:
            score = 1.0
            passed = True

        return CheckResult(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            reasoning=", ".join(reasons),
            cost=0.0,
            latency_ms=0,
            metric_name="readability",
            input_preview=output[:200],
        )

    def sentence_count(self, output: str, min_sentences: int | None = None, max_sentences: int | None = None) -> CheckResult:
        """Count sentences and check against bounds."""
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        count = len(sentences)
        passed = True
        reasons = [f"Sentence count: {count}"]
        if min_sentences is not None:
            if count < min_sentences:
                passed = False
            reasons.append(f"min: {min_sentences}")
        if max_sentences is not None:
            if count > max_sentences:
                passed = False
            reasons.append(f"max: {max_sentences}")

        if min_sentences is not None and max_sentences is not None:
            target = (min_sentences + max_sentences) / 2
            score = max(0.0, 1.0 - abs(count - target) / max(target, 1))
        elif max_sentences is not None:
            score = min(1.0, max_sentences / max(count, 1))
        elif min_sentences is not None:
            score = min(1.0, count / max(min_sentences, 1))
        else:
            score = 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=", ".join(reasons),
            cost=0.0,
            latency_ms=0,
            metric_name="sentence_count",
            input_preview=output[:200],
        )

    # --- Compound checks ---

    def all_of(self, output: str, substrings: list[str]) -> CheckResult:
        """Check that ALL substrings are present in the output."""
        found = [s for s in substrings if s in output]
        missing = [s for s in substrings if s not in output]
        passed = len(missing) == 0
        score = len(found) / max(len(substrings), 1)
        if missing:
            reasoning = f"Missing: {', '.join(repr(s) for s in missing)}"
        else:
            reasoning = f"All {len(substrings)} substrings found"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="all_of",
            input_preview=output[:200],
        )

    def any_of(self, output: str, substrings: list[str]) -> CheckResult:
        """Check that at least ONE substring is present in the output."""
        found = [s for s in substrings if s in output]
        passed = len(found) > 0
        score = min(1.0, len(found) / max(len(substrings), 1))
        if found:
            reasoning = f"Found: {', '.join(repr(s) for s in found)}"
        else:
            reasoning = f"None of {len(substrings)} substrings found"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="any_of",
            input_preview=output[:200],
        )

    def none_of(self, output: str, substrings: list[str]) -> CheckResult:
        """Check that NONE of the substrings are present in the output."""
        found = [s for s in substrings if s in output]
        passed = len(found) == 0
        score = 1.0 - (len(found) / max(len(substrings), 1))
        if found:
            reasoning = f"Unexpectedly found: {', '.join(repr(s) for s in found)}"
        else:
            reasoning = f"None of {len(substrings)} substrings found (good)"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="none_of",
            input_preview=output[:200],
        )

    # --- Code / structure validation ---

    def is_json(self, output: str) -> CheckResult:
        """Check that the output is valid JSON (without requiring a schema)."""
        try:
            json.loads(output)
            return CheckResult(
                passed=True, score=1.0, reasoning="Valid JSON",
                cost=0.0, latency_ms=0, metric_name="is_json",
                input_preview=output[:200],
            )
        except json.JSONDecodeError as e:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Invalid JSON: {e}",
                cost=0.0, latency_ms=0, metric_name="is_json",
                input_preview=output[:200],
            )

    def is_valid_python(self, output: str) -> CheckResult:
        """Check that the output is syntactically valid Python code."""
        # Strip common markdown code fences
        code = output.strip()
        if code.startswith("```python"):
            code = code[len("```python"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        try:
            compile(code, "<checkllm>", "exec")
            return CheckResult(
                passed=True, score=1.0, reasoning="Valid Python syntax",
                cost=0.0, latency_ms=0, metric_name="is_valid_python",
                input_preview=output[:200],
            )
        except SyntaxError as e:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Syntax error: {e.msg} (line {e.lineno})",
                cost=0.0, latency_ms=0, metric_name="is_valid_python",
                input_preview=output[:200],
            )

    # --- PII detection ---

    _PII_PATTERNS: list[tuple[str, str]] = [
        ("email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        ("phone_us", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        ("ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ]

    def no_pii(self, output: str, patterns: list[str] | None = None) -> CheckResult:
        """Check that the output contains no personally identifiable information.

        By default checks for: email, phone, SSN, credit card, IP address.
        Pass ``patterns`` to check only specific types.
        """
        checks = self._PII_PATTERNS
        if patterns:
            checks = [(name, pat) for name, pat in checks if name in patterns]

        found: list[str] = []
        for name, pattern in checks:
            matches = re.findall(pattern, output)
            if matches:
                found.append(f"{name} ({len(matches)})")

        passed = len(found) == 0
        if found:
            reasoning = f"PII detected: {', '.join(found)}"
        else:
            reasoning = f"No PII detected (checked {len(checks)} patterns)"

        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=0,
            metric_name="no_pii",
            input_preview=output[:200],
        )

    # --- Language detection (heuristic) ---

    _COMMON_WORDS: dict[str, set[str]] = {
        "en": {"the", "is", "and", "of", "to", "in", "a", "that", "it", "for", "was", "with", "as", "are", "be", "this", "have", "from", "not", "but"},
        "es": {"el", "la", "de", "en", "y", "que", "es", "un", "los", "del", "las", "por", "con", "una", "para", "no", "se", "su", "al", "lo"},
        "fr": {"le", "la", "de", "et", "en", "un", "une", "est", "que", "les", "des", "du", "dans", "qui", "pour", "pas", "sur", "ce", "par", "avec"},
        "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "ist", "ein", "eine", "dem", "nicht", "auch", "es", "ich"},
        "pt": {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos"},
    }

    def language(self, output: str, expected: str) -> CheckResult:
        """Check that the output appears to be in the expected language.

        Uses word-frequency heuristics. Supports: en, es, fr, de, pt.
        """
        expected = expected.lower()
        if expected not in self._COMMON_WORDS:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Unsupported language '{expected}'. Supported: {', '.join(sorted(self._COMMON_WORDS))}",
                cost=0.0, latency_ms=0, metric_name="language",
                input_preview=output[:200],
            )

        words = set(re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", output.lower()))
        if not words:
            return CheckResult(
                passed=False, score=0.0, reasoning="No words found in output",
                cost=0.0, latency_ms=0, metric_name="language",
                input_preview=output[:200],
            )

        scores: dict[str, float] = {}
        for lang, common in self._COMMON_WORDS.items():
            overlap = len(words & common)
            scores[lang] = overlap / len(common)

        detected = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[expected]
        passed = detected == expected

        return CheckResult(
            passed=passed,
            score=min(1.0, confidence * 5),  # Scale so ~20% overlap = 1.0
            reasoning=f"Detected: {detected} (confidence: {scores[detected]:.2f}), expected: {expected}",
            cost=0.0,
            latency_ms=0,
            metric_name="language",
            input_preview=output[:200],
        )

    # --- Numeric comparison ---

    def _extract_number(self, text: str) -> float | None:
        """Extract the first number from text."""
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group())
        return None

    def greater_than(self, output: str, threshold: float) -> CheckResult:
        """Check that the first number in the output is greater than threshold."""
        value = self._extract_number(output)
        if value is None:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="No number found in output",
                cost=0.0, latency_ms=0, metric_name="greater_than",
                input_preview=output[:200],
            )
        passed = value > threshold
        score = min(1.0, value / max(abs(threshold), 0.001)) if passed else max(0.0, value / max(abs(threshold), 0.001))
        return CheckResult(
            passed=passed, score=min(1.0, max(0.0, score)),
            reasoning=f"Value: {value}, must be > {threshold}",
            cost=0.0, latency_ms=0, metric_name="greater_than",
            input_preview=output[:200],
        )

    def less_than(self, output: str, threshold: float) -> CheckResult:
        """Check that the first number in the output is less than threshold."""
        value = self._extract_number(output)
        if value is None:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="No number found in output",
                cost=0.0, latency_ms=0, metric_name="less_than",
                input_preview=output[:200],
            )
        passed = value < threshold
        score = min(1.0, threshold / max(abs(value), 0.001)) if passed else max(0.0, threshold / max(abs(value), 0.001))
        return CheckResult(
            passed=passed, score=min(1.0, max(0.0, score)),
            reasoning=f"Value: {value}, must be < {threshold}",
            cost=0.0, latency_ms=0, metric_name="less_than",
            input_preview=output[:200],
        )

    def between(self, output: str, low: float, high: float) -> CheckResult:
        """Check that the first number in the output is between low and high (inclusive)."""
        value = self._extract_number(output)
        if value is None:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="No number found in output",
                cost=0.0, latency_ms=0, metric_name="between",
                input_preview=output[:200],
            )
        passed = low <= value <= high
        if passed:
            score = 1.0
        else:
            dist = min(abs(value - low), abs(value - high))
            span = high - low if high > low else 1.0
            score = max(0.0, 1.0 - dist / span)
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"Value: {value}, must be in [{low}, {high}]",
            cost=0.0, latency_ms=0, metric_name="between",
            input_preview=output[:200],
        )

    # --- Text similarity metrics ---

    def bleu(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        """BLEU score (Bilingual Evaluation Understudy) with unigram through 4-gram precision and brevity penalty."""
        output_tokens = output.lower().split()
        reference_tokens = reference.lower().split()

        if not output_tokens or not reference_tokens:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty output or reference",
                cost=0.0, latency_ms=0, metric_name="bleu",
                input_preview=output[:200],
                threshold=threshold,
            )

        # Compute modified n-gram precisions for n=1..4
        log_avg = 0.0
        max_n = min(4, len(output_tokens))
        if max_n == 0:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Output too short to compute BLEU",
                cost=0.0, latency_ms=0, metric_name="bleu",
                input_preview=output[:200],
                threshold=threshold,
            )

        for n in range(1, max_n + 1):
            out_ngrams: Counter[tuple[str, ...]] = Counter(
                tuple(output_tokens[i:i + n]) for i in range(len(output_tokens) - n + 1)
            )
            ref_ngrams: Counter[tuple[str, ...]] = Counter(
                tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)
            )
            clipped = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in out_ngrams.items()}
            numerator = sum(clipped.values())
            denominator = sum(out_ngrams.values())
            if denominator == 0 or numerator == 0:
                # If any n-gram precision is 0, BLEU is 0
                score = 0.0
                passed = score >= threshold
                return CheckResult(
                    passed=passed, score=score,
                    reasoning=f"BLEU: {score:.4f} (threshold: {threshold}) — zero {n}-gram matches",
                    cost=0.0, latency_ms=0, metric_name="bleu",
                    input_preview=output[:200],
                    threshold=threshold,
                )
            log_avg += (1.0 / max_n) * math.log(numerator / denominator)

        # Brevity penalty
        bp = 1.0
        if len(output_tokens) < len(reference_tokens):
            bp = math.exp(1.0 - len(reference_tokens) / len(output_tokens))

        score = bp * math.exp(log_avg)
        score = min(1.0, max(0.0, score))
        passed = score >= threshold
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"BLEU: {score:.4f} (threshold: {threshold}, BP: {bp:.4f})",
            cost=0.0, latency_ms=0, metric_name="bleu",
            input_preview=output[:200],
            threshold=threshold,
        )

    def rouge_l(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        """ROUGE-L score based on Longest Common Subsequence at word level."""
        output_tokens = output.lower().split()
        reference_tokens = reference.lower().split()

        if not output_tokens or not reference_tokens:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty output or reference",
                cost=0.0, latency_ms=0, metric_name="rouge_l",
                input_preview=output[:200],
                threshold=threshold,
            )

        # Compute LCS length using dynamic programming
        m, n = len(output_tokens), len(reference_tokens)
        # Use space-optimized DP (two rows)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if output_tokens[i - 1] == reference_tokens[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        lcs_len = prev[n]

        precision = lcs_len / m if m > 0 else 0.0
        recall = lcs_len / n if n > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        score = min(1.0, max(0.0, f1))
        passed = score >= threshold
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"ROUGE-L F1: {score:.4f} (P: {precision:.4f}, R: {recall:.4f}, threshold: {threshold})",
            cost=0.0, latency_ms=0, metric_name="rouge_l",
            input_preview=output[:200],
            threshold=threshold,
        )

    # --- JSON field-level assertion ---

    def json_field(self, output: str, field_path: str, expected: Any = None, condition: str | None = None) -> CheckResult:
        """JSON field-level assertion with dot-notation navigation and condition support."""
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Invalid JSON: {e}",
                cost=0.0, latency_ms=0, metric_name="json_field",
                input_preview=output[:200],
            )

        # Navigate to field using dot notation
        parts = field_path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return CheckResult(
                        passed=False, score=0.0,
                        reasoning=f"Field path '{field_path}' not found — invalid list index '{part}'",
                        cost=0.0, latency_ms=0, metric_name="json_field",
                        input_preview=output[:200],
                    )
            else:
                return CheckResult(
                    passed=False, score=0.0,
                    reasoning=f"Field path '{field_path}' not found at segment '{part}'",
                    cost=0.0, latency_ms=0, metric_name="json_field",
                    input_preview=output[:200],
                )

        # Evaluate condition or check equality
        if condition is not None:
            return self._eval_json_condition(current, condition, field_path)

        if expected is not None:
            passed = current == expected
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}' = {current!r}, expected {expected!r}",
                cost=0.0, latency_ms=0, metric_name="json_field",
                input_preview=output[:200],
            )

        # No condition and no expected — just check that the field exists
        return CheckResult(
            passed=True, score=1.0,
            reasoning=f"Field '{field_path}' exists with value: {current!r}",
            cost=0.0, latency_ms=0, metric_name="json_field",
            input_preview=output[:200],
        )

    def _eval_json_condition(self, value: Any, condition: str, field_path: str) -> CheckResult:
        """Evaluate a condition against a JSON field value."""
        if condition == "exists":
            # If we got here, the field exists
            return CheckResult(
                passed=True, score=1.0,
                reasoning=f"Field '{field_path}' exists",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        elif condition == "not_empty":
            if value is None:
                passed = False
            elif isinstance(value, (str, list, dict)):
                passed = len(value) > 0
            else:
                passed = True  # Numbers, booleans are considered not empty
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}' {'is not empty' if passed else 'is empty'}: {value!r}",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        elif condition.startswith("gt:"):
            threshold = float(condition[3:])
            try:
                num_val = float(value)
            except (TypeError, ValueError):
                return CheckResult(
                    passed=False, score=0.0,
                    reasoning=f"Field '{field_path}' value {value!r} is not numeric (condition: {condition})",
                    cost=0.0, latency_ms=0, metric_name="json_field",
                )
            passed = num_val > threshold
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}' = {num_val}, {'>' if passed else '<='} {threshold}",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        elif condition.startswith("lt:"):
            threshold = float(condition[3:])
            try:
                num_val = float(value)
            except (TypeError, ValueError):
                return CheckResult(
                    passed=False, score=0.0,
                    reasoning=f"Field '{field_path}' value {value!r} is not numeric (condition: {condition})",
                    cost=0.0, latency_ms=0, metric_name="json_field",
                )
            passed = num_val < threshold
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}' = {num_val}, {'<' if passed else '>='} {threshold}",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        elif condition.startswith("contains:"):
            substr = condition[9:]
            str_val = str(value)
            passed = substr in str_val
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}': '{substr}' {'found' if passed else 'not found'} in {str_val!r}",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        elif condition.startswith("type:"):
            expected_type = condition[5:]
            type_map = {
                "str": str, "string": str,
                "int": int, "integer": int,
                "float": float,
                "bool": bool, "boolean": bool,
                "list": list, "array": list,
                "dict": dict, "object": dict,
                "none": type(None), "null": type(None),
            }
            target_type = type_map.get(expected_type)
            if target_type is None:
                return CheckResult(
                    passed=False, score=0.0,
                    reasoning=f"Unknown type '{expected_type}' in condition",
                    cost=0.0, latency_ms=0, metric_name="json_field",
                )
            passed = isinstance(value, target_type)
            return CheckResult(
                passed=passed, score=1.0 if passed else 0.0,
                reasoning=f"Field '{field_path}' type is {type(value).__name__}, expected {expected_type}",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )
        else:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Unknown condition: '{condition}'",
                cost=0.0, latency_ms=0, metric_name="json_field",
            )

    # --- SQL validation ---

    def is_valid_sql(self, output: str) -> CheckResult:
        """Basic SQL syntax validation without executing."""
        # Strip markdown code fences
        code = output.strip()
        if code.startswith("```sql"):
            code = code[len("```sql"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        if not code:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty SQL statement",
                cost=0.0, latency_ms=0, metric_name="is_valid_sql",
                input_preview=output[:200],
            )

        errors: list[str] = []

        # Check balanced parentheses
        depth = 0
        for ch in code:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                errors.append("Unbalanced parentheses: extra closing ')'")
                break
        if depth > 0:
            errors.append(f"Unbalanced parentheses: {depth} unclosed '('")

        # Check that it starts with a known SQL keyword
        sql_start_keywords = {
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            "WITH", "MERGE", "REPLACE", "TRUNCATE", "EXPLAIN", "DESCRIBE",
            "SHOW", "USE", "SET", "BEGIN", "COMMIT", "ROLLBACK", "GRANT",
            "REVOKE", "EXEC", "EXECUTE", "CALL",
        }
        # Get the first word (uppercase)
        first_word_match = re.match(r'\s*(\w+)', code, re.IGNORECASE)
        if first_word_match:
            first_word = first_word_match.group(1).upper()
            if first_word not in sql_start_keywords:
                errors.append(f"Statement does not start with a recognized SQL keyword (found '{first_word}')")
        else:
            errors.append("Could not identify a SQL keyword at the start")

        # Check for unclosed string literals
        in_single = False
        in_double = False
        for i, ch in enumerate(code):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
        if in_single:
            errors.append("Unclosed single-quoted string literal")
        if in_double:
            errors.append("Unclosed double-quoted string literal")

        if errors:
            return CheckResult(
                passed=False, score=max(0.0, 1.0 - len(errors) * 0.25),
                reasoning=f"SQL validation errors: {'; '.join(errors)}",
                cost=0.0, latency_ms=0, metric_name="is_valid_sql",
                input_preview=output[:200],
            )

        return CheckResult(
            passed=True, score=1.0,
            reasoning="Valid SQL syntax (basic validation passed)",
            cost=0.0, latency_ms=0, metric_name="is_valid_sql",
            input_preview=output[:200],
        )

    # --- Markdown validation ---

    def is_valid_markdown(
        self,
        output: str,
        require_headers: bool = False,
        require_lists: bool = False,
        require_code_blocks: bool = False,
    ) -> CheckResult:
        """Check for valid markdown structure and optionally require specific elements."""
        if not output.strip():
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty output",
                cost=0.0, latency_ms=0, metric_name="is_valid_markdown",
                input_preview=output[:200],
            )

        errors: list[str] = []
        elements_found: list[str] = []

        # Check for headers
        has_headers = bool(re.search(r'^#{1,6}\s+\S', output, re.MULTILINE))
        if has_headers:
            elements_found.append("headers")
        if require_headers and not has_headers:
            errors.append("No headers found (expected at least one)")

        # Check for lists (unordered or ordered)
        has_unordered = bool(re.search(r'^[\s]*[-*+]\s+\S', output, re.MULTILINE))
        has_ordered = bool(re.search(r'^[\s]*\d+\.\s+\S', output, re.MULTILINE))
        has_lists = has_unordered or has_ordered
        if has_lists:
            elements_found.append("lists")
        if require_lists and not has_lists:
            errors.append("No lists found (expected at least one)")

        # Check for code blocks (fenced)
        has_code_blocks = bool(re.search(r'^```', output, re.MULTILINE))
        if has_code_blocks:
            elements_found.append("code blocks")
            # Check that code blocks are properly closed
            fence_count = len(re.findall(r'^```', output, re.MULTILINE))
            if fence_count % 2 != 0:
                errors.append("Unclosed code block (odd number of ``` fences)")
        if require_code_blocks and not has_code_blocks:
            errors.append("No code blocks found (expected at least one)")

        passed = len(errors) == 0
        total_checks = 1 + int(require_headers) + int(require_lists) + int(require_code_blocks)
        passed_checks = total_checks - len(errors)
        score = passed_checks / total_checks

        if errors:
            reasoning = f"Markdown issues: {'; '.join(errors)}"
        else:
            found_str = ", ".join(elements_found) if elements_found else "plain text"
            reasoning = f"Valid markdown structure (elements: {found_str})"

        return CheckResult(
            passed=passed, score=score,
            reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="is_valid_markdown",
            input_preview=output[:200],
        )

    def icontains(self, output: str, substring: str) -> CheckResult:
        """Case-insensitive substring check.

        Args:
            output: The text to search in.
            substring: The substring to look for (case-insensitive).

        Returns:
            CheckResult with pass/fail based on case-insensitive presence.
        """
        found = substring.lower() in output.lower()
        return CheckResult(
            passed=found,
            score=1.0 if found else 0.0,
            reasoning=f"Substring '{substring}' (case-insensitive) {'found' if found else 'not found'} in output",
            cost=0.0, latency_ms=0, metric_name="icontains",
            input_preview=output[:200],
        )

    def icontains_any(self, output: str, substrings: list[str]) -> CheckResult:
        """Case-insensitive check that at least one substring is present.

        Args:
            output: The text to search in.
            substrings: List of substrings to look for (case-insensitive).

        Returns:
            CheckResult with pass/fail based on case-insensitive match of any substring.
        """
        output_lower = output.lower()
        found = [s for s in substrings if s.lower() in output_lower]
        passed = len(found) > 0
        score = min(1.0, len(found) / max(len(substrings), 1))
        if found:
            reasoning = f"Found (case-insensitive): {', '.join(repr(s) for s in found)}"
        else:
            reasoning = f"None of {len(substrings)} substrings found (case-insensitive)"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="icontains_any",
            input_preview=output[:200],
        )

    def icontains_all(self, output: str, substrings: list[str]) -> CheckResult:
        """Case-insensitive check that all substrings are present.

        Args:
            output: The text to search in.
            substrings: List of substrings that must all be present (case-insensitive).

        Returns:
            CheckResult with pass/fail based on case-insensitive match of all substrings.
        """
        output_lower = output.lower()
        found = [s for s in substrings if s.lower() in output_lower]
        missing = [s for s in substrings if s.lower() not in output_lower]
        passed = len(missing) == 0
        score = len(found) / max(len(substrings), 1)
        if missing:
            reasoning = f"Missing (case-insensitive): {', '.join(repr(s) for s in missing)}"
        else:
            reasoning = f"All {len(substrings)} substrings found (case-insensitive)"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="icontains_all",
            input_preview=output[:200],
        )

    def is_html(self, output: str) -> CheckResult:
        """Check that output is valid HTML with balanced tags.

        Looks for common HTML tags and validates that opening/closing tags
        are balanced. Does not require a full DOM parser.

        Args:
            output: The text to validate as HTML.

        Returns:
            CheckResult with pass/fail based on HTML validity.
        """
        text = output.strip()
        if not text:
            return CheckResult(
                passed=False, score=0.0, reasoning="Empty output",
                cost=0.0, latency_ms=0, metric_name="is_html",
                input_preview=output[:200],
            )

        html_tag_pattern = re.compile(r'<(/?)(\w+)(?:\s[^>]*)?\s*(/?)>')
        tags = html_tag_pattern.findall(text)
        if not tags:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="No HTML tags found in output",
                cost=0.0, latency_ms=0, metric_name="is_html",
                input_preview=output[:200],
            )

        void_elements = {
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        }
        stack: list[str] = []
        for is_closing, tag_name, is_self_closing in tags:
            tag_lower = tag_name.lower()
            if is_self_closing or tag_lower in void_elements:
                continue
            if is_closing:
                if not stack or stack[-1] != tag_lower:
                    return CheckResult(
                        passed=False, score=0.0,
                        reasoning=f"Unbalanced HTML: unexpected closing </{tag_lower}>",
                        cost=0.0, latency_ms=0, metric_name="is_html",
                        input_preview=output[:200],
                    )
                stack.pop()
            else:
                stack.append(tag_lower)

        if stack:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Unclosed HTML tags: {', '.join(stack)}",
                cost=0.0, latency_ms=0, metric_name="is_html",
                input_preview=output[:200],
            )

        return CheckResult(
            passed=True, score=1.0,
            reasoning=f"Valid HTML with {len(tags)} tags, all balanced",
            cost=0.0, latency_ms=0, metric_name="is_html",
            input_preview=output[:200],
        )

    def contains_html(self, output: str) -> CheckResult:
        """Check that output contains HTML elements.

        Args:
            output: The text to search for HTML elements.

        Returns:
            CheckResult with pass/fail based on HTML element presence.
        """
        html_pattern = re.compile(r'<(\w+)(?:\s[^>]*)?\s*/?>|</(\w+)\s*>')
        matches = html_pattern.findall(output)
        found = len(matches) > 0
        return CheckResult(
            passed=found,
            score=1.0 if found else 0.0,
            reasoning=f"{'Found' if found else 'No'} HTML elements in output ({len(matches)} tags)",
            cost=0.0, latency_ms=0, metric_name="contains_html",
            input_preview=output[:200],
        )

    def is_xml(self, output: str) -> CheckResult:
        """Check that output is valid XML using xml.etree.ElementTree.

        Args:
            output: The text to validate as XML.

        Returns:
            CheckResult with pass/fail based on XML validity.
        """
        text = output.strip()
        if not text:
            return CheckResult(
                passed=False, score=0.0, reasoning="Empty output",
                cost=0.0, latency_ms=0, metric_name="is_xml",
                input_preview=output[:200],
            )
        try:
            ET.fromstring(text)
            return CheckResult(
                passed=True, score=1.0,
                reasoning="Valid XML document",
                cost=0.0, latency_ms=0, metric_name="is_xml",
                input_preview=output[:200],
            )
        except ET.ParseError as e:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Invalid XML: {e}",
                cost=0.0, latency_ms=0, metric_name="is_xml",
                input_preview=output[:200],
            )

    def contains_xml(self, output: str) -> CheckResult:
        """Check that output contains XML elements.

        Args:
            output: The text to search for XML elements.

        Returns:
            CheckResult with pass/fail based on XML element presence.
        """
        xml_pattern = re.compile(r'<(\w+)(?:\s[^>]*)?>.*?</\1>', re.DOTALL)
        match = xml_pattern.search(output)
        found = match is not None
        return CheckResult(
            passed=found,
            score=1.0 if found else 0.0,
            reasoning=f"XML elements {'found' if found else 'not found'} in output",
            cost=0.0, latency_ms=0, metric_name="contains_xml",
            input_preview=output[:200],
        )

    _REFUSAL_PATTERNS: list[str] = [
        r"I cannot\b",
        r"I can't\b",
        r"I'm sorry,?\s+but I can't",
        r"I'm unable to\b",
        r"I am unable to\b",
        r"I'm not able to\b",
        r"As an AI\b",
        r"As a language model\b",
        r"I don't have the ability\b",
        r"I must decline\b",
        r"I will not\b",
        r"I won't\b",
        r"I refuse to\b",
        r"I apologize,?\s+but I cannot",
        r"I'm afraid I can't",
        r"not appropriate for me to\b",
        r"against my guidelines\b",
        r"violates my (guidelines|policies|rules)",
        r"outside my capabilities\b",
    ]

    def is_refusal(self, output: str) -> CheckResult:
        """Detect if the model refused the request.

        Checks for common refusal phrases like "I cannot", "As an AI", etc.

        Args:
            output: The model output to check.

        Returns:
            CheckResult where passed=True means a refusal WAS detected.
        """
        matched: list[str] = []
        for pattern in self._REFUSAL_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                matched.append(pattern)

        is_refusal = len(matched) > 0
        if is_refusal:
            reasoning = f"Refusal detected ({len(matched)} pattern(s) matched)"
        else:
            reasoning = "No refusal patterns detected"
        return CheckResult(
            passed=is_refusal,
            score=1.0 if is_refusal else 0.0,
            reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="is_refusal",
            input_preview=output[:200],
        )

    def levenshtein(self, output: str, reference: str, threshold: float = 0.7) -> CheckResult:
        """Levenshtein edit distance check with normalized score.

        Computes the Levenshtein similarity ratio (0-1) and passes if it
        meets or exceeds the threshold.

        Args:
            output: The model output.
            reference: The reference text to compare against.
            threshold: Minimum similarity ratio to pass (0.0-1.0).

        Returns:
            CheckResult with score as the similarity ratio.
        """
        ratio = _levenshtein_ratio(output, reference)
        passed = ratio >= threshold
        return CheckResult(
            passed=passed, score=ratio,
            reasoning=f"Levenshtein ratio: {ratio:.4f} (threshold: {threshold})",
            cost=0.0, latency_ms=0, metric_name="levenshtein",
            input_preview=output[:200],
            threshold=threshold,
        )

    def meteor(self, output: str, reference: str, threshold: float = 0.5) -> CheckResult:
        """METEOR-like score using unigram matching with stemming approximation.

        Computes precision, recall, and F-mean with a simple Porter-like
        suffix stripping for stem matching. Includes a fragmentation penalty.

        Args:
            output: The model output.
            reference: The reference text to compare against.
            threshold: Minimum score to pass (0.0-1.0).

        Returns:
            CheckResult with METEOR-like score.
        """
        def simple_stem(word: str) -> str:
            """Approximate English stemming by stripping common suffixes."""
            w = word.lower()
            for suffix in ("ing", "tion", "sion", "ment", "ness", "able", "ible",
                           "ful", "less", "ous", "ive", "ly", "ed", "er", "es", "s"):
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    return w[:-len(suffix)]
            return w

        out_tokens = [simple_stem(w) for w in re.findall(r'\w+', output.lower())]
        ref_tokens = [simple_stem(w) for w in re.findall(r'\w+', reference.lower())]

        if not out_tokens or not ref_tokens:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty output or reference for METEOR computation",
                cost=0.0, latency_ms=0, metric_name="meteor",
                input_preview=output[:200],
                threshold=threshold,
            )

        ref_matched = [False] * len(ref_tokens)
        out_matched = [False] * len(out_tokens)

        for i, ot in enumerate(out_tokens):
            for j, rt in enumerate(ref_tokens):
                if not ref_matched[j] and ot == rt:
                    out_matched[i] = True
                    ref_matched[j] = True
                    break

        matches = sum(out_matched)
        precision = matches / len(out_tokens) if out_tokens else 0.0
        recall = matches / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            score = 0.0
        else:
            alpha = 0.9
            f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

            chunks = 0
            in_chunk = False
            for i in range(len(out_tokens)):
                if out_matched[i]:
                    if not in_chunk:
                        chunks += 1
                        in_chunk = True
                else:
                    in_chunk = False

            frag_penalty = 0.5 * (chunks / max(matches, 1)) if matches > 0 else 0.0
            score = f_mean * (1.0 - frag_penalty)
            score = max(0.0, min(1.0, score))

        passed = score >= threshold
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"METEOR: {score:.4f} (P: {precision:.4f}, R: {recall:.4f}, threshold: {threshold})",
            cost=0.0, latency_ms=0, metric_name="meteor",
            input_preview=output[:200],
            threshold=threshold,
        )

    def perplexity_check(self, output: str, max_perplexity: float = 50.0) -> CheckResult:
        """Simple perplexity proxy using token repetition and vocabulary diversity.

        Higher repetition and lower vocabulary diversity yield a higher
        pseudo-perplexity score. This is a heuristic, not true perplexity.

        Args:
            output: The model output to analyze.
            max_perplexity: Maximum allowable pseudo-perplexity score.

        Returns:
            CheckResult with pseudo-perplexity score.
        """
        tokens = re.findall(r'\w+', output.lower())
        if not tokens:
            return CheckResult(
                passed=True, score=1.0,
                reasoning="Empty output, perplexity not applicable",
                cost=0.0, latency_ms=0, metric_name="perplexity_check",
                input_preview=output[:200],
            )

        total = len(tokens)
        unique = len(set(tokens))
        vocab_diversity = unique / total

        token_counts = Counter(tokens)
        max_freq = max(token_counts.values())
        repetition_ratio = max_freq / total

        pseudo_perplexity = (1.0 / max(vocab_diversity, 0.01)) * (1.0 + repetition_ratio * 10.0)

        passed = pseudo_perplexity <= max_perplexity
        score = min(1.0, max_perplexity / max(pseudo_perplexity, 0.01)) if not passed else 1.0
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"Pseudo-perplexity: {pseudo_perplexity:.2f} (max: {max_perplexity}, vocab diversity: {vocab_diversity:.3f})",
            cost=0.0, latency_ms=0, metric_name="perplexity_check",
            input_preview=output[:200],
        )

    def is_valid_yaml(self, output: str) -> CheckResult:
        """Check that output is valid YAML using yaml.safe_load.

        Args:
            output: The text to validate as YAML.

        Returns:
            CheckResult with pass/fail based on YAML validity.
        """
        import yaml

        text = output.strip()
        if text.startswith("```yaml"):
            text = text[len("```yaml"):]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if not text:
            return CheckResult(
                passed=False, score=0.0, reasoning="Empty output",
                cost=0.0, latency_ms=0, metric_name="is_valid_yaml",
                input_preview=output[:200],
            )
        try:
            result = yaml.safe_load(text)
            if result is None:
                return CheckResult(
                    passed=False, score=0.0,
                    reasoning="YAML parsed as None/empty",
                    cost=0.0, latency_ms=0, metric_name="is_valid_yaml",
                    input_preview=output[:200],
                )
            return CheckResult(
                passed=True, score=1.0,
                reasoning=f"Valid YAML (parsed as {type(result).__name__})",
                cost=0.0, latency_ms=0, metric_name="is_valid_yaml",
                input_preview=output[:200],
            )
        except yaml.YAMLError as e:
            return CheckResult(
                passed=False, score=0.0,
                reasoning=f"Invalid YAML: {e}",
                cost=0.0, latency_ms=0, metric_name="is_valid_yaml",
                input_preview=output[:200],
            )

    _CITATION_PATTERNS: list[str] = [
        r'\[\d+\]',
        r'\(\w[\w\s&.,]+,\s*\d{4}\)',
        r'\(\w[\w\s&.,]+\s+\d{4}\)',
        r'\b(?:doi|DOI):\s*\S+',
        r'https?://\S+',
    ]

    def has_citations(self, output: str, min_count: int = 1) -> CheckResult:
        """Check that output contains citation patterns.

        Looks for patterns like [1], (Author, Year), DOI references, and URLs.

        Args:
            output: The text to search for citations.
            min_count: Minimum number of citations required.

        Returns:
            CheckResult with pass/fail based on citation count.
        """
        all_matches: list[str] = []
        for pattern in self._CITATION_PATTERNS:
            all_matches.extend(re.findall(pattern, output))

        count = len(all_matches)
        passed = count >= min_count
        score = min(1.0, count / max(min_count, 1))
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"Found {count} citation(s), minimum required: {min_count}",
            cost=0.0, latency_ms=0, metric_name="has_citations",
            input_preview=output[:200],
        )

    def no_repetition(self, output: str, max_ngram_repeat: int = 3) -> CheckResult:
        """Detect excessive n-gram repetition in output.

        Checks for repeated sequences of 3-grams or longer that appear
        more than max_ngram_repeat times, which is a sign of degenerate output.

        Args:
            output: The text to check for repetition.
            max_ngram_repeat: Maximum allowed repetitions of any n-gram (n>=3).

        Returns:
            CheckResult where passed=True means no excessive repetition found.
        """
        tokens = output.lower().split()
        if len(tokens) < 3:
            return CheckResult(
                passed=True, score=1.0,
                reasoning="Output too short for repetition analysis",
                cost=0.0, latency_ms=0, metric_name="no_repetition",
                input_preview=output[:200],
            )

        worst_ratio = 0.0
        worst_ngram = ""
        for n in (3, 4, 5):
            if len(tokens) < n:
                continue
            ngrams: Counter[tuple[str, ...]] = Counter(
                tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            )
            for ng, count in ngrams.most_common(1):
                if count > max_ngram_repeat:
                    ratio = count / (len(tokens) - n + 1)
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_ngram = " ".join(ng)

        if worst_ngram:
            passed = False
            score = max(0.0, 1.0 - worst_ratio)
            reasoning = f"Excessive repetition: '{worst_ngram}' repeated too many times (ratio: {worst_ratio:.3f})"
        else:
            passed = True
            score = 1.0
            reasoning = f"No excessive n-gram repetition detected (max allowed: {max_ngram_repeat})"

        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="no_repetition",
            input_preview=output[:200],
        )

    def semantic_similarity(self, output: str, reference: str, threshold: float = 0.7) -> CheckResult:
        """Cosine similarity using TF-IDF vectors.

        Computes term frequency-inverse document frequency vectors for
        both texts and measures their cosine similarity. No external
        model is needed.

        Args:
            output: The model output.
            reference: The reference text to compare against.
            threshold: Minimum cosine similarity to pass (0.0-1.0).

        Returns:
            CheckResult with cosine similarity score.
        """
        from scipy.spatial.distance import cosine as cosine_dist

        def tokenize(text: str) -> list[str]:
            return re.findall(r'\w+', text.lower())

        out_tokens = tokenize(output)
        ref_tokens = tokenize(reference)

        if not out_tokens or not ref_tokens:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="Empty output or reference for similarity computation",
                cost=0.0, latency_ms=0, metric_name="semantic_similarity",
                input_preview=output[:200],
                threshold=threshold,
            )

        all_tokens = sorted(set(out_tokens + ref_tokens))
        token_to_idx = {t: i for i, t in enumerate(all_tokens)}
        vocab_size = len(all_tokens)

        out_counts = Counter(out_tokens)
        ref_counts = Counter(ref_tokens)

        out_tf = [0.0] * vocab_size
        ref_tf = [0.0] * vocab_size
        for token, count in out_counts.items():
            out_tf[token_to_idx[token]] = count / len(out_tokens)
        for token, count in ref_counts.items():
            ref_tf[token_to_idx[token]] = count / len(ref_tokens)

        num_docs = 2
        idf = [0.0] * vocab_size
        for i, token in enumerate(all_tokens):
            doc_freq = int(token in out_counts) + int(token in ref_counts)
            idf[i] = math.log((num_docs + 1) / (doc_freq + 1)) + 1

        out_tfidf = [tf * idf_val for tf, idf_val in zip(out_tf, idf)]
        ref_tfidf = [tf * idf_val for tf, idf_val in zip(ref_tf, idf)]

        out_norm = math.sqrt(sum(v * v for v in out_tfidf))
        ref_norm = math.sqrt(sum(v * v for v in ref_tfidf))

        if out_norm == 0 or ref_norm == 0:
            cos_sim = 0.0
        else:
            cos_sim = 1.0 - cosine_dist(out_tfidf, ref_tfidf)

        cos_sim = max(0.0, min(1.0, cos_sim))
        passed = cos_sim >= threshold
        return CheckResult(
            passed=passed, score=cos_sim,
            reasoning=f"TF-IDF cosine similarity: {cos_sim:.4f} (threshold: {threshold})",
            cost=0.0, latency_ms=0, metric_name="semantic_similarity",
            input_preview=output[:200],
            threshold=threshold,
        )

    def is_valid_url(self, output: str) -> CheckResult:
        """Check if output contains valid URL patterns.

        Args:
            output: The text to check for valid URLs.

        Returns:
            CheckResult with pass/fail based on URL presence and validity.
        """
        url_pattern = re.compile(
            r'https?://[^\s<>"\')\]]+|'
            r'ftp://[^\s<>"\')\]]+|'
            r'www\.[^\s<>"\')\]]+\.[a-zA-Z]{2,}'
        )
        matches = url_pattern.findall(output)
        if not matches:
            return CheckResult(
                passed=False, score=0.0,
                reasoning="No valid URL patterns found in output",
                cost=0.0, latency_ms=0, metric_name="is_valid_url",
                input_preview=output[:200],
            )

        valid_urls: list[str] = []
        for url in matches:
            if not url.startswith(("http://", "https://", "ftp://")):
                url = "http://" + url
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc and "." in parsed.netloc:
                valid_urls.append(url)

        passed = len(valid_urls) > 0
        score = len(valid_urls) / max(len(matches), 1)
        return CheckResult(
            passed=passed, score=score,
            reasoning=f"Found {len(valid_urls)} valid URL(s) out of {len(matches)} candidates",
            cost=0.0, latency_ms=0, metric_name="is_valid_url",
            input_preview=output[:200],
        )

    _STRUCTURE_DETECTORS: dict[str, str] = {
        "headers": r'^#{1,6}\s+\S',
        "bullet_points": r'^[\s]*[-*+]\s+\S',
        "numbered_lists": r'^[\s]*\d+\.\s+\S',
        "code_blocks": r'^```',
        "blockquotes": r'^>\s+',
        "bold": r'\*\*\S.*?\S\*\*|__\S.*?\S__',
        "italic": r'(?<!\*)\*(?!\*)(?:\S.*?\S|\S)\*(?!\*)|(?<!_)_(?!_)(?:\S.*?\S|\S)_(?!_)',
        "links": r'\[.+?\]\(.+?\)',
        "tables": r'\|.+\|',
    }

    def has_structure(self, output: str, elements: list[str]) -> CheckResult:
        """Check that output has specific structural elements.

        Supported elements: headers, bullet_points, numbered_lists,
        code_blocks, blockquotes, bold, italic, links, tables.

        Args:
            output: The text to check for structure.
            elements: List of structural element names to require.

        Returns:
            CheckResult with pass/fail based on element presence.
        """
        if not elements:
            return CheckResult(
                passed=True, score=1.0,
                reasoning="No structural elements required",
                cost=0.0, latency_ms=0, metric_name="has_structure",
                input_preview=output[:200],
            )

        found: list[str] = []
        missing: list[str] = []
        for elem in elements:
            pattern = self._STRUCTURE_DETECTORS.get(elem)
            if pattern is None:
                missing.append(f"{elem} (unknown element)")
                continue
            if re.search(pattern, output, re.MULTILINE):
                found.append(elem)
            else:
                missing.append(elem)

        passed = len(missing) == 0
        score = len(found) / max(len(elements), 1)
        if missing:
            reasoning = f"Missing structural elements: {', '.join(missing)}"
        else:
            reasoning = f"All {len(elements)} structural elements found: {', '.join(found)}"
        return CheckResult(
            passed=passed, score=score, reasoning=reasoning,
            cost=0.0, latency_ms=0, metric_name="has_structure",
            input_preview=output[:200],
        )
