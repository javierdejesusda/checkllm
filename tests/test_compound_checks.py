"""Tests for compound checks (all_of, any_of, none_of) and code validation."""

from checkllm.deterministic import DeterministicChecks


class TestAllOf:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_all_present(self):
        result = self.checks.all_of(
            "Python is a programming language", ["Python", "programming", "language"]
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_some_missing(self):
        result = self.checks.all_of("Python is great", ["Python", "Java", "Rust"])
        assert result.passed is False
        assert result.score < 1.0
        assert "Java" in result.reasoning
        assert "Rust" in result.reasoning

    def test_all_missing(self):
        result = self.checks.all_of("hello world", ["foo", "bar", "baz"])
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_list(self):
        result = self.checks.all_of("anything", [])
        assert result.passed is True


class TestAnyOf:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_one_found(self):
        result = self.checks.any_of("Python is great", ["Python", "Java", "Rust"])
        assert result.passed is True

    def test_none_found(self):
        result = self.checks.any_of("hello world", ["foo", "bar", "baz"])
        assert result.passed is False
        assert result.score == 0.0

    def test_all_found(self):
        result = self.checks.any_of("Python Java Rust", ["Python", "Java", "Rust"])
        assert result.passed is True
        assert result.score == 1.0


class TestNoneOf:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_none_found(self):
        result = self.checks.none_of("hello world", ["foo", "bar", "baz"])
        assert result.passed is True
        assert result.score == 1.0

    def test_some_found(self):
        result = self.checks.none_of("Python is great", ["Python", "Java"])
        assert result.passed is False
        assert "Python" in result.reasoning

    def test_all_found(self):
        result = self.checks.none_of("foo bar baz", ["foo", "bar", "baz"])
        assert result.passed is False
        assert result.score == 0.0


class TestIsJson:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_valid_json_object(self):
        result = self.checks.is_json('{"key": "value"}')
        assert result.passed is True

    def test_valid_json_array(self):
        result = self.checks.is_json("[1, 2, 3]")
        assert result.passed is True

    def test_valid_json_string(self):
        result = self.checks.is_json('"hello"')
        assert result.passed is True

    def test_invalid_json(self):
        result = self.checks.is_json("not json at all")
        assert result.passed is False
        assert "Invalid JSON" in result.reasoning

    def test_empty_string(self):
        result = self.checks.is_json("")
        assert result.passed is False


class TestIsValidPython:
    def setup_method(self):
        self.checks = DeterministicChecks()

    def test_valid_python(self):
        result = self.checks.is_valid_python("x = 1 + 2\nprint(x)")
        assert result.passed is True

    def test_valid_function(self):
        code = "def hello():\n    return 'world'"
        result = self.checks.is_valid_python(code)
        assert result.passed is True

    def test_invalid_python(self):
        result = self.checks.is_valid_python("def hello(\n    broken")
        assert result.passed is False
        assert "Syntax error" in result.reasoning

    def test_strips_markdown_fences(self):
        code = "```python\nprint('hello')\n```"
        result = self.checks.is_valid_python(code)
        assert result.passed is True

    def test_strips_generic_fences(self):
        code = "```\nx = 42\n```"
        result = self.checks.is_valid_python(code)
        assert result.passed is True
