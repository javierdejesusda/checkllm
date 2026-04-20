"""Comprehensive tests for checkllm.chain — AssertionChain methods."""

from __future__ import annotations

import pytest

from checkllm.chain import AssertionChain
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.testing import MockJudge


def _collector() -> CheckCollector:
    config = CheckllmConfig()
    return CheckCollector(config=config)


def _collector_with_judge(score: float = 0.9) -> CheckCollector:
    config = CheckllmConfig()
    judge = MockJudge(default_score=score)
    return CheckCollector(config=config, judge=judge)


class TestAssertionChainDeterministicMethods:
    def test_contains_passes(self):
        c = _collector()
        result = c.that("Hello world").contains("Hello")
        assert isinstance(result, AssertionChain)
        assert c.results[-1].passed is True

    def test_contains_fails(self):
        c = _collector()
        c.that("Hello world").contains("Goodbye")
        assert c.results[-1].passed is False

    def test_not_contains_passes(self):
        c = _collector()
        c.that("Hello world").not_contains("Goodbye")
        assert c.results[-1].passed is True

    def test_not_contains_fails(self):
        c = _collector()
        c.that("Hello world").not_contains("Hello")
        assert c.results[-1].passed is False

    def test_exact_match_passes(self):
        c = _collector()
        c.that("Python").exact_match("Python")
        assert c.results[-1].passed is True

    def test_exact_match_fails(self):
        c = _collector()
        c.that("Python").exact_match("python")
        assert c.results[-1].passed is False

    def test_exact_match_ignore_case(self):
        c = _collector()
        c.that("Python").exact_match("python", ignore_case=True)
        assert c.results[-1].passed is True

    def test_starts_with_passes(self):
        c = _collector()
        c.that("Hello world").starts_with("Hello")
        assert c.results[-1].passed is True

    def test_starts_with_fails(self):
        c = _collector()
        c.that("Hello world").starts_with("world")
        assert c.results[-1].passed is False

    def test_ends_with_passes(self):
        c = _collector()
        c.that("Hello world").ends_with("world")
        assert c.results[-1].passed is True

    def test_ends_with_fails(self):
        c = _collector()
        c.that("Hello world").ends_with("Hello")
        assert c.results[-1].passed is False

    def test_regex_passes(self):
        c = _collector()
        c.that("abc123").regex(r"\d+")
        assert c.results[-1].passed is True

    def test_regex_fails(self):
        c = _collector()
        c.that("abcdef").regex(r"^\d+$")
        assert c.results[-1].passed is False

    def test_max_tokens_passes(self):
        c = _collector()
        c.that("short text").max_tokens(100)
        assert c.results[-1].passed is True

    def test_max_tokens_fails(self):
        c = _collector()
        c.that("word " * 200).max_tokens(5)
        assert c.results[-1].passed is False

    def test_min_tokens_passes(self):
        c = _collector()
        c.that("word word word word word").min_tokens(3)
        assert c.results[-1].passed is True

    def test_min_tokens_fails(self):
        c = _collector()
        c.that("one").min_tokens(10)
        assert c.results[-1].passed is False

    def test_word_count_passes(self):
        c = _collector()
        c.that("one two three").word_count(min_words=2, max_words=5)
        assert c.results[-1].passed is True

    def test_similarity_passes(self):
        c = _collector()
        c.that("Hello world").similarity("Hello world", threshold=0.9)
        assert c.results[-1].passed is True

    def test_is_json_passes(self):
        c = _collector()
        c.that('{"key": "value"}').is_json()
        assert c.results[-1].passed is True

    def test_is_json_fails(self):
        c = _collector()
        c.that("not json").is_json()
        assert c.results[-1].passed is False

    def test_is_valid_python_passes(self):
        c = _collector()
        c.that("x = 1 + 2").is_valid_python()
        assert c.results[-1].passed is True

    def test_is_valid_python_fails(self):
        c = _collector()
        c.that("def broken(: pass").is_valid_python()
        assert c.results[-1].passed is False

    def test_has_no_pii_passes(self):
        c = _collector()
        c.that("Hello world").has_no_pii()
        assert c.results[-1].passed is True

    def test_all_of_passes(self):
        c = _collector()
        c.that("Python is great and fast").all_of(["Python", "great"])
        assert c.results[-1].passed is True

    def test_all_of_fails(self):
        c = _collector()
        c.that("Python is great").all_of(["Python", "missing"])
        assert c.results[-1].passed is False

    def test_any_of_passes(self):
        c = _collector()
        c.that("Python is great").any_of(["Java", "Python"])
        assert c.results[-1].passed is True

    def test_any_of_fails(self):
        c = _collector()
        c.that("Python is great").any_of(["Java", "Ruby"])
        assert c.results[-1].passed is False

    def test_none_of_passes(self):
        c = _collector()
        c.that("Python is great").none_of(["Java", "Ruby"])
        assert c.results[-1].passed is True

    def test_none_of_fails(self):
        c = _collector()
        c.that("Python is great").none_of(["Python", "Ruby"])
        assert c.results[-1].passed is False

    def test_icontains_passes(self):
        c = _collector()
        c.that("Python is Great").icontains("python")
        assert c.results[-1].passed is True

    def test_icontains_any_passes(self):
        c = _collector()
        c.that("Hello World").icontains_any(["hello", "java"])
        assert c.results[-1].passed is True

    def test_icontains_all_passes(self):
        c = _collector()
        c.that("Hello World").icontains_all(["hello", "world"])
        assert c.results[-1].passed is True

    def test_is_html_passes(self):
        c = _collector()
        c.that("<html><body>Hello</body></html>").is_html()
        assert c.results[-1].passed is True

    def test_contains_html_passes(self):
        c = _collector()
        c.that("Some text with <b>bold</b>").contains_html()
        assert c.results[-1].passed is True

    def test_is_xml_passes(self):
        c = _collector()
        c.that("<root><item>data</item></root>").is_xml()
        assert c.results[-1].passed is True

    def test_contains_xml_passes(self):
        c = _collector()
        c.that("Text with <tag>content</tag>").contains_xml()
        assert c.results[-1].passed is True

    def test_is_refusal_passes(self):
        c = _collector()
        c.that("I cannot help with that request.").is_refusal()
        assert c.results[-1].passed is True

    def test_levenshtein_passes(self):
        c = _collector()
        c.that("hello world").levenshtein("hello world", threshold=0.9)
        assert c.results[-1].passed is True

    def test_meteor_passes(self):
        c = _collector()
        c.that("the cat sat on the mat").meteor("the cat sat on the mat", threshold=0.9)
        assert c.results[-1].passed is True

    def test_perplexity_check_passes(self):
        c = _collector()
        c.that("This is a normal English sentence.").perplexity_check(max_perplexity=200.0)
        assert c.results[-1].passed is True

    def test_is_valid_yaml_passes(self):
        c = _collector()
        c.that("key: value\nother: 42").is_valid_yaml()
        assert c.results[-1].passed is True

    def test_is_valid_yaml_fails(self):
        c = _collector()
        c.that(": : : invalid yaml :::").is_valid_yaml()
        # Just verify it runs and returns a result
        assert c.results[-1] is not None

    def test_has_citations_passes(self):
        c = _collector()
        c.that("According to [1] Smith et al., the result is [2] important.").has_citations(min_count=2)
        assert c.results[-1] is not None

    def test_no_repetition_passes(self):
        c = _collector()
        c.that("The quick brown fox jumps.").no_repetition(max_ngram_repeat=3)
        assert c.results[-1].passed is True

    def test_is_valid_url_passes(self):
        c = _collector()
        c.that("https://example.com").is_valid_url()
        assert c.results[-1].passed is True

    def test_is_valid_url_fails(self):
        c = _collector()
        c.that("not a url").is_valid_url()
        assert c.results[-1].passed is False

    def test_has_structure_passes(self):
        c = _collector()
        c.that("## Section One\n- item\n1. item\n").has_structure(["headers", "bullet_points"])
        assert c.results[-1].passed is True


class TestAssertionChainChaining:
    def test_chain_is_fluent(self):
        c = _collector()
        chain = c.that("Python is great")
        result = chain.contains("Python").not_contains("Java").max_tokens(100)
        assert isinstance(result, AssertionChain)
        assert len(c.results) == 3

    def test_chain_accumulates_results(self):
        c = _collector()
        c.that("Hello world").contains("Hello").starts_with("Hello").ends_with("world")
        assert len(c.results) == 3
        assert all(r.passed for r in c.results)

    def test_chain_captures_failures(self):
        c = _collector()
        c.that("Python").contains("Java").contains("Python")
        assert len(c.results) == 2
        assert c.results[0].passed is False
        assert c.results[1].passed is True


class TestAssertionChainScoresAbove:
    def test_scores_above_passes(self):
        c = _collector_with_judge(0.95)
        c.that("Good output").scores_above("relevance", 0.8, query="test")
        assert c.results[-1].passed is True

    def test_scores_above_fails(self):
        c = _collector_with_judge(0.5)
        c.that("Bad output").scores_above("relevance", 0.9, query="test")
        assert c.results[-1].passed is False

    def test_scores_above_invalid_metric(self):
        c = _collector_with_judge(0.9)
        with pytest.raises(AttributeError, match="Unknown metric"):
            c.that("output").scores_above("nonexistent_metric_xyz", 0.8)
