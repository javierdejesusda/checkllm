"""Tests for new deterministic checks: similarity, readability, word_count, etc."""

from checkllm.deterministic import DeterministicChecks, _levenshtein_ratio, _flesch_kincaid_grade


class TestLevenshteinRatio:
    def test_identical(self):
        assert _levenshtein_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        assert _levenshtein_ratio("abc", "xyz") < 0.5

    def test_empty_strings(self):
        assert _levenshtein_ratio("", "") == 1.0
        assert _levenshtein_ratio("abc", "") == 0.0
        assert _levenshtein_ratio("", "abc") == 0.0

    def test_one_edit(self):
        ratio = _levenshtein_ratio("hello", "hallo")
        assert 0.7 < ratio < 1.0

    def test_symmetry(self):
        assert _levenshtein_ratio("abc", "abd") == _levenshtein_ratio("abd", "abc")


class TestFleschKincaid:
    def test_simple_sentence(self):
        grade = _flesch_kincaid_grade("The cat sat on the mat.")
        assert grade < 5.0  # Simple sentence = low grade level

    def test_complex_text(self):
        text = (
            "The implementation of sophisticated algorithms requires "
            "comprehensive understanding of computational complexity theory "
            "and mathematical optimization techniques."
        )
        grade = _flesch_kincaid_grade(text)
        assert grade > 10.0  # Complex text = high grade level

    def test_empty(self):
        assert _flesch_kincaid_grade("") == 0.0


class TestSimilarityCheck:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_identical_strings(self):
        result = self.checks.similarity("hello world", "hello world")
        assert result.passed is True
        assert result.score == 1.0

    def test_similar_strings(self):
        result = self.checks.similarity("hello world", "hello worlds", threshold=0.8)
        assert result.passed is True
        assert result.score > 0.8

    def test_different_strings(self):
        result = self.checks.similarity("hello", "completely different text", threshold=0.8)
        assert result.passed is False

    def test_ignore_case(self):
        result = self.checks.similarity("Hello World", "hello world", threshold=0.99, ignore_case=True)
        assert result.passed is True
        assert result.score == 1.0

    def test_custom_threshold(self):
        result = self.checks.similarity("cat", "bat", threshold=0.5)
        assert result.passed is True


class TestMinTokens:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_above_minimum(self):
        result = self.checks.min_tokens("hello world this is a test", minimum=3)
        assert result.passed is True

    def test_below_minimum(self):
        result = self.checks.min_tokens("hi", minimum=100)
        assert result.passed is False


class TestWordCount:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_within_range(self):
        result = self.checks.word_count("one two three four five", min_words=3, max_words=10)
        assert result.passed is True

    def test_below_min(self):
        result = self.checks.word_count("hello", min_words=5)
        assert result.passed is False

    def test_above_max(self):
        result = self.checks.word_count("one two three four five", max_words=3)
        assert result.passed is False

    def test_no_bounds(self):
        result = self.checks.word_count("anything goes here")
        assert result.passed is True
        assert result.score == 1.0


class TestCharCount:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_within_range(self):
        result = self.checks.char_count("hello world", min_chars=5, max_chars=50)
        assert result.passed is True

    def test_below_min(self):
        result = self.checks.char_count("hi", min_chars=10)
        assert result.passed is False

    def test_above_max(self):
        result = self.checks.char_count("hello world", max_chars=5)
        assert result.passed is False


class TestReadability:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_simple_text_passes_low_grade(self):
        result = self.checks.readability("The cat sat on the mat.", max_grade=8.0)
        assert result.passed is True

    def test_complex_text_fails_low_grade(self):
        text = (
            "The implementation of sophisticated computational algorithms "
            "necessitates a comprehensive understanding of advanced mathematical "
            "optimization techniques and theoretical frameworks."
        )
        result = self.checks.readability(text, max_grade=5.0)
        assert result.passed is False

    def test_min_grade(self):
        result = self.checks.readability("Hi. Go. Run.", min_grade=10.0)
        assert result.passed is False

    def test_no_bounds(self):
        result = self.checks.readability("Some text here.")
        assert result.passed is True


class TestSentenceCount:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_within_range(self):
        result = self.checks.sentence_count("One sentence. Two sentences. Three.", min_sentences=2, max_sentences=5)
        assert result.passed is True

    def test_below_min(self):
        result = self.checks.sentence_count("Just one.", min_sentences=3)
        assert result.passed is False

    def test_above_max(self):
        result = self.checks.sentence_count("One. Two. Three. Four. Five.", max_sentences=3)
        assert result.passed is False
