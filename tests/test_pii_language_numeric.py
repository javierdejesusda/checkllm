"""Tests for PII detection, language detection, and numeric comparison checks."""
import pytest

from checkllm.deterministic import DeterministicChecks


class TestNoPii:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_no_pii_clean_text(self):
        result = self.checks.no_pii("This is a clean text with no personal info.")
        assert result.passed is True

    def test_detects_email(self):
        result = self.checks.no_pii("Contact me at john@example.com please")
        assert result.passed is False
        assert "email" in result.reasoning

    def test_detects_phone(self):
        result = self.checks.no_pii("Call me at (555) 123-4567")
        assert result.passed is False
        assert "phone" in result.reasoning

    def test_detects_ssn(self):
        result = self.checks.no_pii("My SSN is 123-45-6789")
        assert result.passed is False
        assert "ssn" in result.reasoning

    def test_detects_credit_card(self):
        result = self.checks.no_pii("My card is 4111-1111-1111-1111")
        assert result.passed is False
        assert "credit_card" in result.reasoning

    def test_detects_ip(self):
        result = self.checks.no_pii("Server at 192.168.1.1")
        assert result.passed is False
        assert "ip_address" in result.reasoning

    def test_filter_patterns(self):
        text = "Email: john@example.com, Phone: 555-123-4567"
        result = self.checks.no_pii(text, patterns=["email"])
        assert result.passed is False
        assert "email" in result.reasoning
        # Phone should not be checked
        assert "phone" not in result.reasoning

    def test_multiple_detections(self):
        text = "Email me at a@b.com or call 555-123-4567"
        result = self.checks.no_pii(text)
        assert result.passed is False


class TestLanguageDetection:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_english_detected(self):
        result = self.checks.language(
            "The quick brown fox jumps over the lazy dog and is a great animal",
            expected="en"
        )
        assert result.passed is True
        assert result.metric_name == "language"

    def test_spanish_detected(self):
        result = self.checks.language(
            "El perro grande corre por el parque con los gatos y las aves",
            expected="es"
        )
        assert result.passed is True

    def test_wrong_language(self):
        result = self.checks.language(
            "The quick brown fox jumps over the lazy dog",
            expected="es"
        )
        assert result.passed is False

    def test_unsupported_language(self):
        result = self.checks.language("test text", expected="zh")
        assert result.passed is False
        assert "Unsupported" in result.reasoning

    def test_empty_text(self):
        result = self.checks.language("", expected="en")
        assert result.passed is False


class TestGreaterThan:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_passes(self):
        result = self.checks.greater_than("The answer is 42", 10)
        assert result.passed is True
        assert "42" in result.reasoning

    def test_fails(self):
        result = self.checks.greater_than("Only 3 items", 10)
        assert result.passed is False

    def test_no_number(self):
        result = self.checks.greater_than("No numbers here", 10)
        assert result.passed is False
        assert "No number" in result.reasoning

    def test_negative_number(self):
        result = self.checks.greater_than("Temperature is -5 degrees", -10)
        assert result.passed is True


class TestLessThan:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_passes(self):
        result = self.checks.less_than("Only 3 items", 10)
        assert result.passed is True

    def test_fails(self):
        result = self.checks.less_than("The answer is 42", 10)
        assert result.passed is False

    def test_no_number(self):
        result = self.checks.less_than("No numbers", 10)
        assert result.passed is False


class TestBetween:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_in_range(self):
        result = self.checks.between("Score: 7.5", 1.0, 10.0)
        assert result.passed is True
        assert result.score == 1.0

    def test_below_range(self):
        result = self.checks.between("Score: 0.5", 1.0, 10.0)
        assert result.passed is False

    def test_above_range(self):
        result = self.checks.between("Score: 15", 1.0, 10.0)
        assert result.passed is False

    def test_boundary_inclusive(self):
        result = self.checks.between("Exactly 10", 1.0, 10.0)
        assert result.passed is True

    def test_no_number(self):
        result = self.checks.between("No numbers", 1.0, 10.0)
        assert result.passed is False
