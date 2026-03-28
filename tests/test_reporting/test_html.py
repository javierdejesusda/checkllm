from pathlib import Path

from checkllm.models import CheckResult
from checkllm.reporting.html import generate_html_report


class TestHtmlReport:
    def test_generates_html_file(self, tmp_path: Path):
        results = {
            "test_summarizer": [
                CheckResult(
                    passed=True, score=0.95, reasoning="All good",
                    cost=0.002, latency_ms=450, metric_name="hallucination",
                ),
            ]
        }
        output = tmp_path / "report.html"
        generate_html_report(results, output)
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_contains_test_results(self, tmp_path: Path):
        results = {
            "test_foo": [
                CheckResult(
                    passed=True, score=0.9, reasoning="ok",
                    cost=0.001, latency_ms=100, metric_name="hallucination",
                ),
                CheckResult(
                    passed=False, score=0.3, reasoning="failed check",
                    cost=0.002, latency_ms=200, metric_name="relevance",
                ),
            ]
        }
        output = tmp_path / "report.html"
        generate_html_report(results, output)
        content = output.read_text()
        assert "test_foo" in content
        assert "hallucination" in content
        assert "relevance" in content
        assert "failed check" in content

    def test_contains_summary_stats(self, tmp_path: Path):
        results = {
            "test_a": [
                CheckResult(passed=True, score=0.9, reasoning="ok", cost=0.003, latency_ms=100, metric_name="h"),
                CheckResult(passed=False, score=0.2, reasoning="bad", cost=0.002, latency_ms=200, metric_name="r"),
            ],
        }
        output = tmp_path / "report.html"
        generate_html_report(results, output)
        content = output.read_text()
        assert "1" in content  # passed count
        assert "1" in content  # failed count
