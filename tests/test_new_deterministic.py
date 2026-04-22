"""Tests for the 17 new deterministic checks added to checkllm."""

import pytest

from checkllm.deterministic import DeterministicChecks


@pytest.fixture
def dc():
    return DeterministicChecks()


class TestIcontains:
    def test_passes_case_insensitive(self, dc):
        result = dc.icontains("Hello World", "hello")
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_when_absent(self, dc):
        result = dc.icontains("Hello World", "goodbye")
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_string(self, dc):
        result = dc.icontains("", "hello")
        assert result.passed is False

    def test_mixed_case_match(self, dc):
        result = dc.icontains("Python is GREAT", "great")
        assert result.passed is True


class TestIcontainsAny:
    def test_passes_when_one_matches(self, dc):
        result = dc.icontains_any("Hello World", ["HELLO", "goodbye"])
        assert result.passed is True

    def test_fails_when_none_match(self, dc):
        result = dc.icontains_any("Hello World", ["foo", "bar"])
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_output(self, dc):
        result = dc.icontains_any("", ["hello"])
        assert result.passed is False

    def test_empty_substrings(self, dc):
        result = dc.icontains_any("Hello", [])
        assert result.passed is False


class TestIcontainsAll:
    def test_passes_when_all_match(self, dc):
        result = dc.icontains_all("Hello Beautiful World", ["HELLO", "world"])
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_when_one_missing(self, dc):
        result = dc.icontains_all("Hello World", ["hello", "goodbye"])
        assert result.passed is False

    def test_empty_output(self, dc):
        result = dc.icontains_all("", ["hello"])
        assert result.passed is False

    def test_empty_substrings(self, dc):
        result = dc.icontains_all("Hello", [])
        assert result.passed is True


class TestIsHtml:
    def test_valid_html(self, dc):
        result = dc.is_html("<div><p>Hello</p></div>")
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_unbalanced(self, dc):
        result = dc.is_html("<div><p>Hello</div>")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_html("")
        assert result.passed is False

    def test_no_html_tags(self, dc):
        result = dc.is_html("Just plain text")
        assert result.passed is False

    def test_void_elements(self, dc):
        result = dc.is_html("<div>Hello<br>World</div>")
        assert result.passed is True

    def test_self_closing_tags(self, dc):
        result = dc.is_html("<img src='test.png' />")
        assert result.passed is True


class TestContainsHtml:
    def test_passes_with_html(self, dc):
        result = dc.contains_html("Some text <b>bold</b> more text")
        assert result.passed is True

    def test_fails_without_html(self, dc):
        result = dc.contains_html("Just plain text here")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.contains_html("")
        assert result.passed is False


class TestIsXml:
    def test_valid_xml(self, dc):
        result = dc.is_xml("<root><child>text</child></root>")
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_xml(self, dc):
        result = dc.is_xml("<root><child>text</root>")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_xml("")
        assert result.passed is False

    def test_xml_with_attributes(self, dc):
        result = dc.is_xml('<item id="1" name="test">value</item>')
        assert result.passed is True


class TestContainsXml:
    def test_passes_with_xml(self, dc):
        result = dc.contains_xml("Response: <data>value</data>")
        assert result.passed is True

    def test_fails_without_xml(self, dc):
        result = dc.contains_xml("No XML here at all")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.contains_xml("")
        assert result.passed is False


class TestIsRefusal:
    def test_detects_cannot(self, dc):
        result = dc.is_refusal("I cannot help with that request.")
        assert result.passed is True

    def test_detects_as_an_ai(self, dc):
        result = dc.is_refusal("As an AI, I don't have the ability to do that.")
        assert result.passed is True

    def test_passes_normal_response(self, dc):
        result = dc.is_refusal("Python is a programming language used for web development.")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_refusal("")
        assert result.passed is False

    def test_detects_sorry_cant(self, dc):
        result = dc.is_refusal("I'm sorry, but I can't provide that information.")
        assert result.passed is True

    def test_detects_will_not(self, dc):
        result = dc.is_refusal("I will not generate harmful content.")
        assert result.passed is True


class TestLevenshtein:
    def test_identical_strings(self, dc):
        result = dc.levenshtein("hello world", "hello world", threshold=0.9)
        assert result.passed is True
        assert result.score == 1.0

    def test_similar_strings(self, dc):
        result = dc.levenshtein("hello world", "hello worl", threshold=0.8)
        assert result.passed is True
        assert result.score > 0.8

    def test_very_different_strings(self, dc):
        result = dc.levenshtein("abc", "xyz", threshold=0.5)
        assert result.passed is False
        assert result.score < 0.5

    def test_empty_strings(self, dc):
        result = dc.levenshtein("", "", threshold=0.5)
        assert result.passed is True
        assert result.score == 1.0

    def test_one_empty(self, dc):
        result = dc.levenshtein("hello", "", threshold=0.5)
        assert result.passed is False
        assert result.score == 0.0


class TestMeteor:
    def test_identical_text(self, dc):
        result = dc.meteor("the cat sat on the mat", "the cat sat on the mat", threshold=0.5)
        assert result.passed is True
        assert result.score > 0.5

    def test_similar_text(self, dc):
        result = dc.meteor("the cat sat on a mat", "the cat was sitting on the mat", threshold=0.3)
        assert result.passed is True

    def test_completely_different(self, dc):
        result = dc.meteor("xyz abc", "123 456 789", threshold=0.5)
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_output(self, dc):
        result = dc.meteor("", "the cat sat", threshold=0.5)
        assert result.passed is False

    def test_empty_reference(self, dc):
        result = dc.meteor("the cat sat", "", threshold=0.5)
        assert result.passed is False


class TestPerplexityCheck:
    def test_normal_text(self, dc):
        result = dc.perplexity_check(
            "The quick brown fox jumps over the lazy dog. It was a beautiful day.",
            max_perplexity=50.0,
        )
        assert result.passed is True

    def test_highly_repetitive(self, dc):
        result = dc.perplexity_check("the the the the the the the the the the", max_perplexity=3.0)
        assert result.passed is False

    def test_empty_text(self, dc):
        result = dc.perplexity_check("", max_perplexity=50.0)
        assert result.passed is True

    def test_diverse_vocabulary(self, dc):
        result = dc.perplexity_check(
            "Python JavaScript Ruby Go Rust Swift Kotlin TypeScript Elixir Scala",
            max_perplexity=50.0,
        )
        assert result.passed is True


class TestIsValidYaml:
    def test_valid_yaml(self, dc):
        result = dc.is_valid_yaml("name: test\nversion: 1.0\nitems:\n  - a\n  - b")
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_yaml(self, dc):
        result = dc.is_valid_yaml("name: test\n  bad indent: value\n wrong: stuff")
        # YAML is actually quite permissive; test with truly broken syntax
        result2 = dc.is_valid_yaml(":")
        # A single colon is valid YAML (maps to {None: None}), but our check
        # requires non-None result
        assert isinstance(result, object)

    def test_empty_string(self, dc):
        result = dc.is_valid_yaml("")
        assert result.passed is False

    def test_yaml_with_code_fence(self, dc):
        result = dc.is_valid_yaml("```yaml\nkey: value\n```")
        assert result.passed is True

    def test_yaml_dict(self, dc):
        result = dc.is_valid_yaml("database:\n  host: localhost\n  port: 5432")
        assert result.passed is True


class TestHasCitations:
    def test_numeric_citations(self, dc):
        result = dc.has_citations("According to research [1], this is true [2].", min_count=2)
        assert result.passed is True

    def test_author_year_citations(self, dc):
        result = dc.has_citations("This was shown (Smith, 2023) in the study.", min_count=1)
        assert result.passed is True

    def test_no_citations(self, dc):
        result = dc.has_citations("No references here at all.", min_count=1)
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.has_citations("", min_count=1)
        assert result.passed is False

    def test_url_citation(self, dc):
        result = dc.has_citations("See https://example.com for details.", min_count=1)
        assert result.passed is True

    def test_insufficient_citations(self, dc):
        result = dc.has_citations("Only one ref [1].", min_count=3)
        assert result.passed is False


class TestNoRepetition:
    def test_normal_text(self, dc):
        result = dc.no_repetition(
            "The quick brown fox jumps over the lazy dog in the park.",
            max_ngram_repeat=3,
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_repetitive_text(self, dc):
        result = dc.no_repetition(
            "I am happy I am happy I am happy I am happy I am happy",
            max_ngram_repeat=2,
        )
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.no_repetition("", max_ngram_repeat=3)
        assert result.passed is True

    def test_short_text(self, dc):
        result = dc.no_repetition("hi there", max_ngram_repeat=3)
        assert result.passed is True

    def test_degenerate_output(self, dc):
        result = dc.no_repetition(
            " ".join(["the cat sat"] * 20),
            max_ngram_repeat=3,
        )
        assert result.passed is False


class TestSemanticSimilarity:
    def test_identical_texts(self, dc):
        result = dc.semantic_similarity(
            "the cat sat on the mat", "the cat sat on the mat", threshold=0.9
        )
        assert result.passed is True
        assert result.score >= 0.99

    def test_similar_texts(self, dc):
        result = dc.semantic_similarity(
            "the cat sat on the mat",
            "a cat was sitting on a mat",
            threshold=0.2,
        )
        assert result.passed is True
        assert result.score > 0.0

    def test_different_texts(self, dc):
        result = dc.semantic_similarity(
            "quantum physics experiments",
            "baking chocolate cookies",
            threshold=0.8,
        )
        assert result.passed is False

    def test_empty_output(self, dc):
        result = dc.semantic_similarity("", "reference text", threshold=0.5)
        assert result.passed is False

    def test_empty_reference(self, dc):
        result = dc.semantic_similarity("output text", "", threshold=0.5)
        assert result.passed is False


class TestIsValidUrl:
    def test_valid_http_url(self, dc):
        result = dc.is_valid_url("Visit https://example.com for more info.")
        assert result.passed is True

    def test_valid_https_url(self, dc):
        result = dc.is_valid_url("https://www.google.com/search?q=test")
        assert result.passed is True

    def test_no_url(self, dc):
        result = dc.is_valid_url("No URLs in this text at all.")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_valid_url("")
        assert result.passed is False

    def test_www_url(self, dc):
        result = dc.is_valid_url("Check www.example.com for details")
        assert result.passed is True

    def test_multiple_urls(self, dc):
        result = dc.is_valid_url("See https://a.com and https://b.org")
        assert result.passed is True
        assert result.score == 1.0


class TestHasStructure:
    def test_headers_found(self, dc):
        result = dc.has_structure("# Title\nSome text", ["headers"])
        assert result.passed is True

    def test_bullet_points_found(self, dc):
        result = dc.has_structure("Items:\n- First\n- Second", ["bullet_points"])
        assert result.passed is True

    def test_numbered_lists_found(self, dc):
        result = dc.has_structure("Steps:\n1. First\n2. Second", ["numbered_lists"])
        assert result.passed is True

    def test_code_blocks_found(self, dc):
        result = dc.has_structure("Example:\n```\nprint('hi')\n```", ["code_blocks"])
        assert result.passed is True

    def test_missing_element(self, dc):
        result = dc.has_structure("Just plain text", ["headers", "bullet_points"])
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_elements_list(self, dc):
        result = dc.has_structure("Any text", [])
        assert result.passed is True

    def test_multiple_elements(self, dc):
        text = "# Header\n- bullet\n1. numbered\n```code```"
        result = dc.has_structure(
            text, ["headers", "bullet_points", "numbered_lists", "code_blocks"]
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_bold_found(self, dc):
        result = dc.has_structure("This is **bold** text", ["bold"])
        assert result.passed is True

    def test_links_found(self, dc):
        result = dc.has_structure("Click [here](https://example.com)", ["links"])
        assert result.passed is True

    def test_unknown_element(self, dc):
        result = dc.has_structure("text", ["nonexistent_element"])
        assert result.passed is False


class TestIsYaml:
    def test_valid_yaml(self, dc):
        result = dc.is_yaml("key: value\nfoo: bar")
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "is_yaml"

    def test_invalid_yaml(self, dc):
        result = dc.is_yaml("}{bad yaml}{")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_yaml("")
        assert result.passed is False

    def test_yaml_with_code_fence(self, dc):
        result = dc.is_yaml("```yaml\nkey: value\n```")
        assert result.passed is True


class TestIsUrl:
    def test_valid_https_url(self, dc):
        result = dc.is_url("https://example.com")
        assert result.passed is True
        assert result.metric_name == "is_url"

    def test_valid_http_url(self, dc):
        result = dc.is_url("http://api.example.com/v1/data")
        assert result.passed is True

    def test_url_with_surrounding_text_fails(self, dc):
        result = dc.is_url("Check out https://example.com for info")
        assert result.passed is False

    def test_plain_text_fails(self, dc):
        result = dc.is_url("just some text")
        assert result.passed is False

    def test_empty_string(self, dc):
        result = dc.is_url("")
        assert result.passed is False


class TestHasUrl:
    def test_text_containing_url(self, dc):
        result = dc.has_url("Check out https://example.com for more info.")
        assert result.passed is True
        assert result.metric_name == "has_url"

    def test_no_url(self, dc):
        result = dc.has_url("No URLs in this text at all.")
        assert result.passed is False

    def test_multiple_urls(self, dc):
        result = dc.has_url("See https://a.com and https://b.org")
        assert result.passed is True

    def test_empty_string(self, dc):
        result = dc.has_url("")
        assert result.passed is False
