"""Tests for Markdown report generation."""


from checkllm.models import CheckResult
from checkllm.reporting.markdown import generate_markdown_report


def _sample_results():
    return {
        "test_foo": [
            CheckResult(passed=True, score=0.9, reasoning="Good", cost=0.001, latency_ms=100, metric_name="hallucination"),
            CheckResult(passed=False, score=0.3, reasoning="Bad output", cost=0.002, latency_ms=200, metric_name="relevance"),
        ],
        "test_bar": [
            CheckResult(passed=True, score=1.0, reasoning="Perfect", cost=0.0, latency_ms=0, metric_name="contains"),
        ],
    }


class TestMarkdownReport:
    def test_generates_string(self):
        md = generate_markdown_report(_sample_results())
        assert "# checkllm Report" in md
        assert "test_foo" in md
        assert "test_bar" in md
        assert "PASS" in md
        assert "FAIL" in md

    def test_summary_line(self):
        md = generate_markdown_report(_sample_results())
        assert "2/3" in md  # 2 passed out of 3

    def test_writes_to_file(self, tmp_path):
        output = tmp_path / "report.md"
        generate_markdown_report(_sample_results(), output)
        assert output.exists()
        content = output.read_text()
        assert "checkllm Report" in content

    def test_table_format(self):
        md = generate_markdown_report(_sample_results())
        assert "| Status | Metric |" in md
        assert "hallucination" in md
