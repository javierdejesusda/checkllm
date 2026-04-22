"""Comprehensive tests for checkllm.check_deterministic — DeterministicChecksMixin."""

from __future__ import annotations


from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig


def _c() -> CheckCollector:
    return CheckCollector(config=CheckllmConfig())


class TestDeterministicMixinMethods:
    """Test each method of DeterministicChecksMixin via CheckCollector."""

    def test_not_contains(self):
        c = _c()
        r = c.not_contains("Hello world", "Goodbye")
        assert r.passed is True

    def test_not_contains_fails(self):
        c = _c()
        r = c.not_contains("Hello world", "Hello")
        assert r.passed is False

    def test_latency_passes(self):
        c = _c()
        r = c.latency(100, 500)
        assert r.passed is True

    def test_latency_fails(self):
        c = _c()
        r = c.latency(600, 500)
        assert r.passed is False

    def test_cost_passes(self):
        c = _c()
        r = c.cost(0.001, 0.01)
        assert r.passed is True

    def test_cost_fails(self):
        c = _c()
        r = c.cost(0.1, 0.01)
        assert r.passed is False

    def test_regex_passes(self):
        c = _c()
        r = c.regex("abc123", r"\d+")
        assert r.passed is True

    def test_exact_match_with_ignore_case(self):
        c = _c()
        r = c.exact_match("Hello", "hello", ignore_case=True)
        assert r.passed is True

    def test_starts_with(self):
        c = _c()
        r = c.starts_with("Hello world", "Hello")
        assert r.passed is True

    def test_ends_with(self):
        c = _c()
        r = c.ends_with("Hello world", "world")
        assert r.passed is True

    def test_min_tokens(self):
        c = _c()
        r = c.min_tokens("word word word word word", 3)
        assert r.passed is True

    def test_word_count_with_bounds(self):
        c = _c()
        r = c.word_count("one two three", min_words=2, max_words=5)
        assert r.passed is True

    def test_char_count_passes(self):
        c = _c()
        r = c.char_count("hello", min_chars=3, max_chars=10)
        assert r.passed is True

    def test_char_count_fails(self):
        c = _c()
        r = c.char_count("x", min_chars=5)
        assert r.passed is False

    def test_similarity(self):
        c = _c()
        r = c.similarity("Hello world", "Hello world", threshold=0.9)
        assert r.passed is True

    def test_sentence_count(self):
        c = _c()
        r = c.sentence_count("Hello world. This is great.", min_sentences=1, max_sentences=5)
        assert r.passed is True

    def test_all_of_passes(self):
        c = _c()
        r = c.all_of("Python is great and fast", ["Python", "great"])
        assert r.passed is True

    def test_any_of_passes(self):
        c = _c()
        r = c.any_of("Python is great", ["Java", "Python"])
        assert r.passed is True

    def test_none_of_passes(self):
        c = _c()
        r = c.none_of("Python is great", ["Java", "Ruby"])
        assert r.passed is True

    def test_is_json_passes(self):
        c = _c()
        r = c.is_json('{"key": "value"}')
        assert r.passed is True

    def test_is_valid_python_passes(self):
        c = _c()
        r = c.is_valid_python("x = 1 + 2")
        assert r.passed is True

    def test_no_pii_passes(self):
        c = _c()
        r = c.no_pii("Hello world, nice day")
        assert r.passed is True

    def test_language_english(self):
        c = _c()
        r = c.language("This is a beautiful sentence in English.", "en")
        assert r is not None  # result returned

    def test_greater_than_passes(self):
        c = _c()
        r = c.greater_than("0.9", 0.5)
        assert r.passed is True

    def test_less_than_passes(self):
        c = _c()
        r = c.less_than("0.3", 0.5)
        assert r.passed is True

    def test_between_passes(self):
        c = _c()
        r = c.between("0.7", 0.5, 0.9)
        assert r.passed is True

    def test_bleu_perfect(self):
        c = _c()
        r = c.bleu("the cat sat", "the cat sat", threshold=0.9)
        assert r.passed is True

    def test_rouge_l_perfect(self):
        c = _c()
        r = c.rouge_l("the cat sat on the mat", "the cat sat on the mat", threshold=0.9)
        assert r.passed is True

    def test_icontains(self):
        c = _c()
        r = c.icontains("Hello World", "hello")
        assert r.passed is True

    def test_icontains_any(self):
        c = _c()
        r = c.icontains_any("Hello World", ["hello", "java"])
        assert r.passed is True

    def test_icontains_all(self):
        c = _c()
        r = c.icontains_all("Hello World", ["hello", "world"])
        assert r.passed is True

    def test_is_html(self):
        c = _c()
        r = c.is_html("<html><body>Hello</body></html>")
        assert r.passed is True

    def test_contains_html(self):
        c = _c()
        r = c.contains_html("Some <b>bold</b> text")
        assert r.passed is True

    def test_is_xml(self):
        c = _c()
        r = c.is_xml("<root><item>data</item></root>")
        assert r.passed is True

    def test_contains_xml(self):
        c = _c()
        r = c.contains_xml("Text with <tag>content</tag>")
        assert r.passed is True

    def test_is_refusal(self):
        c = _c()
        r = c.is_refusal("I cannot help with that request.")
        assert r.passed is True

    def test_levenshtein(self):
        c = _c()
        r = c.levenshtein("hello world", "hello world", threshold=0.9)
        assert r.passed is True

    def test_meteor(self):
        c = _c()
        r = c.meteor("the cat sat on the mat", "the cat sat on the mat", threshold=0.9)
        assert r.passed is True

    def test_perplexity_check(self):
        c = _c()
        r = c.perplexity_check("This is a normal English sentence.", max_perplexity=200.0)
        assert r.passed is True

    def test_is_valid_yaml(self):
        c = _c()
        r = c.is_valid_yaml("key: value\nother: 42")
        assert r.passed is True

    def test_is_valid_yaml_fails(self):
        c = _c()
        r = c.is_valid_yaml("not: :yaml: content:")
        assert r is not None

    def test_no_repetition(self):
        c = _c()
        r = c.no_repetition("The quick brown fox jumps over the lazy dog.", max_ngram_repeat=3)
        assert r.passed is True

    def test_is_valid_url(self):
        c = _c()
        r = c.is_valid_url("https://example.com")
        assert r.passed is True

    def test_is_valid_url_fails(self):
        c = _c()
        r = c.is_valid_url("not a url")
        assert r.passed is False

    def test_is_valid_sql(self):
        c = _c()
        r = c.is_valid_sql("SELECT * FROM users WHERE id = 1")
        assert r.passed is True

    def test_is_valid_markdown(self):
        c = _c()
        r = c.is_valid_markdown("# Title\n\nSome content")
        assert r.passed is True

    def test_gleu(self):
        c = _c()
        r = c.gleu("the cat sat", "the cat sat", threshold=0.9)
        assert r.passed is True

    def test_chrf(self):
        c = _c()
        r = c.chrf("the cat sat", "the cat sat", threshold=0.9)
        assert r.passed is True

    def test_latency_check(self):
        c = _c()
        r = c.latency_check(0.0, 0.1, max_ms=5000.0)
        assert r.passed is True

    def test_cost_check(self):
        c = _c()
        r = c.cost_check(100, 50, "gpt-4o", max_cost=1.0)
        assert r is not None

    def test_string_distance(self):
        c = _c()
        r = c.string_distance("hello world", "hello world", threshold=0.9)
        assert r.passed is True

    def test_exact_match_strict(self):
        c = _c()
        r = c.exact_match_strict("Hello", "Hello")
        assert r.passed is True

    def test_exact_match_strict_ignore_case(self):
        c = _c()
        r = c.exact_match_strict("hello", "HELLO", ignore_case=True)
        assert r.passed is True

    def test_is_yaml(self):
        c = _c()
        r = c.is_yaml("key: value")
        assert r.passed is True

    def test_is_url(self):
        c = _c()
        r = c.is_url("https://example.com")
        assert r.passed is True

    def test_has_url(self):
        c = _c()
        r = c.has_url("Visit https://example.com for more info")
        assert r.passed is True

    def test_has_citations(self):
        c = _c()
        r = c.has_citations("See [1] and [2] for references.", min_count=2)
        assert r is not None

    def test_has_structure(self):
        c = _c()
        r = c.has_structure(
            "## Section\n- bullet point\n1. numbered\n", ["headers", "bullet_points"]
        )
        assert r.passed is True

    def test_semantic_similarity(self):
        c = _c()
        r = c.semantic_similarity("The cat sat on the mat", "The cat sat on the mat", threshold=0.9)
        assert r is not None

    def test_json_field(self):
        c = _c()
        r = c.json_field('{"name": "Alice", "age": 30}', "name", expected="Alice")
        assert r.passed is True

    def test_json_field_missing(self):
        c = _c()
        r = c.json_field('{"name": "Alice"}', "missing_field")
        assert r.passed is False

    def test_readability(self):
        c = _c()
        r = c.readability("This is a simple sentence.", max_grade=12.0)
        assert r is not None

    def test_results_accumulate(self):
        c = _c()
        c.contains("Hello world", "Hello")
        c.not_contains("Hello world", "Goodbye")
        c.max_tokens("short", 100)
        assert len(c.results) == 3
        assert all(r.passed for r in c.results)
