"""Tests for all new production-readiness features."""
import json


from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks


class TestExactMatch:
    def test_passes_on_match(self):
        dc = DeterministicChecks()
        result = dc.exact_match("hello world", "hello world")
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_on_mismatch(self):
        dc = DeterministicChecks()
        result = dc.exact_match("hello world", "goodbye world")
        assert result.passed is False

    def test_strips_whitespace(self):
        dc = DeterministicChecks()
        result = dc.exact_match("  hello  ", "hello")
        assert result.passed is True

    def test_case_insensitive(self):
        dc = DeterministicChecks()
        result = dc.exact_match("Hello World", "hello world", ignore_case=True)
        assert result.passed is True

    def test_case_sensitive_by_default(self):
        dc = DeterministicChecks()
        result = dc.exact_match("Hello", "hello")
        assert result.passed is False


class TestStartsWith:
    def test_passes(self):
        dc = DeterministicChecks()
        result = dc.starts_with("Hello world", "Hello")
        assert result.passed is True

    def test_fails(self):
        dc = DeterministicChecks()
        result = dc.starts_with("Hello world", "World")
        assert result.passed is False


class TestEndsWith:
    def test_passes(self):
        dc = DeterministicChecks()
        result = dc.ends_with("Hello world", "world")
        assert result.passed is True

    def test_fails(self):
        dc = DeterministicChecks()
        result = dc.ends_with("Hello world", "Hello")
        assert result.passed is False


class TestCheckCollectorNewChecks:
    def test_exact_match_on_collector(self):
        c = CheckCollector(config=CheckllmConfig())
        c.exact_match("42", "42")
        assert c.results[0].passed is True
        assert c.results[0].metric_name == "exact_match"

    def test_starts_with_on_collector(self):
        c = CheckCollector(config=CheckllmConfig())
        c.starts_with("Python is great", "Python")
        assert c.results[0].passed is True

    def test_ends_with_on_collector(self):
        c = CheckCollector(config=CheckllmConfig())
        c.ends_with("Python is great", "great")
        assert c.results[0].passed is True


class TestSnapshotDeduplication:
    def test_duplicate_metric_names_preserved(self, pytester, tmp_path):
        """Two contains checks should both appear in snapshot with indexed keys."""
        snapshot_path = tmp_path / "snap.json"
        pytester.makepyfile(
            """
            def test_two_contains(check):
                check.contains("hello world", "hello")
                check.contains("hello world", "world")
            """
        )
        pytester.runpytest_subprocess("-v", f"--checkllm-snapshot={snapshot_path}")
        assert snapshot_path.exists()

        data = json.loads(snapshot_path.read_text())
        tests = data["tests"]
        node_id = list(tests.keys())[0]
        metrics = tests[node_id][0]["metrics"]
        # Should have "contains" and "contains_1", not just "contains"
        assert "contains" in metrics
        assert "contains_1" in metrics
        assert metrics["contains"]["score"] == 1.0
        assert metrics["contains_1"]["score"] == 1.0


class TestGracefulApiKeySkip:
    def test_llm_check_skips_without_api_key(self, pytester, monkeypatch):
        """LLM checks should skip (not crash) when no API key is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        pytester.makepyfile(
            """
            def test_needs_api_key(check):
                check.hallucination("output", context="context")
            """
        )
        result = pytester.runpytest_subprocess("-v")
        # Should skip, not error
        stdout = result.stdout.str()
        assert "SKIP" in stdout or "skip" in stdout.lower() or result.ret == 0


class TestCaseExpectedField:
    def test_case_has_expected(self):
        from checkllm.datasets.case import Case

        case = Case(input="What is 2+2?", expected="4", context="math")
        assert case.expected == "4"
        assert case.context == "math"


class TestPytestMarker:
    def test_llm_marker_registered(self, pytester):
        """The @pytest.mark.llm marker should be registered without warnings."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.llm
            def test_marked():
                pass
            """
        )
        result = pytester.runpytest_subprocess("-v", "--strict-markers")
        result.stdout.fnmatch_lines(["*1 passed*"])


class TestCliInit:
    def test_init_creates_complete_project(self, tmp_path):
        from typer.testing import CliRunner
        from checkllm.cli import app

        result = CliRunner().invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0

        # Verify all files created
        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "tests" / "test_llm_example.py").exists()
        assert (tmp_path / "tests" / "fixtures" / "cases.yaml").exists()

        # Verify pyproject.toml has checkllm config
        content = (tmp_path / "pyproject.toml").read_text()
        assert "[tool.checkllm]" in content

        # Verify sample test is valid Python
        test_content = (tmp_path / "tests" / "test_llm_example.py").read_text()
        assert "def test_output_quality" in test_content

        # Verify dataset has expected field
        dataset_content = (tmp_path / "tests" / "fixtures" / "cases.yaml").read_text()
        assert "expected:" in dataset_content
