from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Type

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
                reasoning=f"No number found in output",
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
                reasoning=f"No number found in output",
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
                reasoning=f"No number found in output",
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
