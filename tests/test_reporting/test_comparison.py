"""Tests for A/B comparison reporting module."""
from __future__ import annotations

from pathlib import Path


from checkllm.models import CheckResult
from checkllm.reporting.comparison import (
    ComparisonReport,
    generate_comparison_html,
    generate_comparison_markdown,
    render_comparison_terminal,
)


def _results_a() -> dict[str, list[CheckResult]]:
    return {
        "test_summary": [
            CheckResult(
                passed=True, score=0.85, reasoning="Decent summary",
                cost=0.002, latency_ms=300, metric_name="hallucination",
            ),
            CheckResult(
                passed=True, score=0.9, reasoning="Relevant",
                cost=0.001, latency_ms=200, metric_name="relevance",
            ),
        ],
    }


def _results_b() -> dict[str, list[CheckResult]]:
    return {
        "test_summary": [
            CheckResult(
                passed=True, score=0.95, reasoning="Great summary",
                cost=0.003, latency_ms=250, metric_name="hallucination",
            ),
            CheckResult(
                passed=False, score=0.4, reasoning="Not relevant",
                cost=0.001, latency_ms=180, metric_name="relevance",
            ),
        ],
    }


def _make_report() -> ComparisonReport:
    return ComparisonReport(
        results_a=_results_a(),
        results_b=_results_b(),
        label_a="GPT-4o",
        label_b="Claude-3",
    )


class TestComparisonReport:
    def test_dataclass_fields(self):
        report = _make_report()
        assert report.label_a == "GPT-4o"
        assert report.label_b == "Claude-3"
        assert "test_summary" in report.results_a
        assert "test_summary" in report.results_b


class TestComparisonHtml:
    def test_generates_html_file(self, tmp_path: Path):
        output = tmp_path / "comparison.html"
        generate_comparison_html(_make_report(), output)
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content

    def test_contains_labels(self, tmp_path: Path):
        output = tmp_path / "comparison.html"
        generate_comparison_html(_make_report(), output)
        content = output.read_text()
        assert "GPT-4o" in content
        assert "Claude-3" in content

    def test_contains_metrics(self, tmp_path: Path):
        output = tmp_path / "comparison.html"
        generate_comparison_html(_make_report(), output)
        content = output.read_text()
        assert "hallucination" in content
        assert "relevance" in content

    def test_contains_delta(self, tmp_path: Path):
        output = tmp_path / "comparison.html"
        generate_comparison_html(_make_report(), output)
        content = output.read_text()
        # hallucination: 0.95 - 0.85 = +0.100
        assert "+0.100" in content

    def test_contains_summary_stats(self, tmp_path: Path):
        output = tmp_path / "comparison.html"
        generate_comparison_html(_make_report(), output)
        content = output.read_text()
        assert "Pass rate" in content
        assert "Avg score" in content
        assert "Total cost" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "sub" / "comparison.html"
        generate_comparison_html(_make_report(), output)
        assert output.exists()


class TestComparisonMarkdown:
    def test_generates_markdown(self, tmp_path: Path):
        output = tmp_path / "comparison.md"
        md = generate_comparison_markdown(_make_report(), output)
        assert output.exists()
        assert isinstance(md, str)
        assert "# checkllm Comparison" in md

    def test_contains_labels(self, tmp_path: Path):
        output = tmp_path / "comparison.md"
        md = generate_comparison_markdown(_make_report(), output)
        assert "GPT-4o" in md
        assert "Claude-3" in md

    def test_contains_delta(self, tmp_path: Path):
        output = tmp_path / "comparison.md"
        md = generate_comparison_markdown(_make_report(), output)
        assert "+0.100" in md

    def test_summary_table(self, tmp_path: Path):
        output = tmp_path / "comparison.md"
        md = generate_comparison_markdown(_make_report(), output)
        assert "| Pass rate" in md
        assert "| Avg score" in md
        assert "| Total cost" in md


class TestComparisonTerminal:
    def test_renders_to_string(self):
        output = render_comparison_terminal(_make_report(), to_string=True)
        assert output is not None
        assert "GPT-4o" in output
        assert "Claude-3" in output

    def test_contains_summary(self):
        output = render_comparison_terminal(_make_report(), to_string=True)
        assert "Summary" in output

    def test_contains_delta(self):
        output = render_comparison_terminal(_make_report(), to_string=True)
        assert "Detailed Results" in output

    def test_returns_none_without_to_string(self):
        result = render_comparison_terminal(_make_report(), to_string=False)
        assert result is None


class TestComparisonEdgeCases:
    def test_disjoint_tests(self, tmp_path: Path):
        """A and B have different test names."""
        report = ComparisonReport(
            results_a={
                "test_alpha": [
                    CheckResult(
                        passed=True, score=0.9, reasoning="ok",
                        cost=0.001, latency_ms=100, metric_name="hallucination",
                    ),
                ],
            },
            results_b={
                "test_beta": [
                    CheckResult(
                        passed=True, score=0.8, reasoning="fine",
                        cost=0.002, latency_ms=150, metric_name="relevance",
                    ),
                ],
            },
            label_a="A",
            label_b="B",
        )
        output = tmp_path / "disjoint.html"
        generate_comparison_html(report, output)
        content = output.read_text()
        assert "test_alpha" in content
        assert "test_beta" in content

    def test_tie(self, tmp_path: Path):
        """Both sides have identical scores."""
        shared = {
            "test_x": [
                CheckResult(
                    passed=True, score=0.8, reasoning="ok",
                    cost=0.001, latency_ms=100, metric_name="hallucination",
                ),
            ],
        }
        report = ComparisonReport(
            results_a=shared, results_b=shared,
            label_a="Run1", label_b="Run2",
        )
        output = tmp_path / "tie.html"
        generate_comparison_html(report, output)
        content = output.read_text()
        assert "Tie" in content
