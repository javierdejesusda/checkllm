"""Tests for trend reporting module."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from checkllm.models import CheckResult
from checkllm.reporting.trends import (
    TrendData,
    generate_trend_html,
    render_trend_terminal,
    _sparkline,
)


def _make_trend_data(count: int = 3) -> list[TrendData]:
    """Generate *count* trend snapshots with progressively better scores."""
    data: list[TrendData] = []
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(count):
        score = 0.5 + i * 0.15  # 0.50, 0.65, 0.80
        cost = 0.01 - i * 0.002  # 0.010, 0.008, 0.006
        data.append(
            TrendData(
                run_id=f"run-{i}",
                timestamp=base_time + timedelta(days=i),
                label=f"v{i + 1}",
                results={
                    "test_summary": [
                        CheckResult(
                            passed=score >= 0.6,
                            score=score,
                            reasoning=f"Score {score:.2f}",
                            cost=cost,
                            latency_ms=100 + i * 50,
                            metric_name="hallucination",
                        ),
                    ],
                },
            )
        )
    return data


class TestTrendData:
    def test_pass_rate(self):
        data = _make_trend_data(3)
        assert data[0].pass_rate == 0.0  # score 0.50 => not passed
        assert data[1].pass_rate == 1.0  # score 0.65 => passed
        assert data[2].pass_rate == 1.0  # score 0.80 => passed

    def test_avg_score(self):
        data = _make_trend_data(3)
        assert abs(data[0].avg_score - 0.50) < 0.01
        assert abs(data[1].avg_score - 0.65) < 0.01
        assert abs(data[2].avg_score - 0.80) < 0.01

    def test_total_cost(self):
        data = _make_trend_data(3)
        assert abs(data[0].total_cost - 0.010) < 0.0001
        assert abs(data[2].total_cost - 0.006) < 0.0001

    def test_metric_avg(self):
        data = _make_trend_data(1)
        assert data[0].metric_avg("hallucination") == pytest.approx(0.5, abs=0.01)
        assert data[0].metric_avg("nonexistent") is None


class TestSparkline:
    def test_basic(self):
        result = _sparkline([0.0, 0.5, 1.0])
        assert len(result) == 3

    def test_all_same(self):
        result = _sparkline([0.5, 0.5, 0.5])
        assert len(result) == 3
        # All the same value => all same char
        assert result[0] == result[1] == result[2]

    def test_empty(self):
        assert _sparkline([]) == ""

    def test_single_value(self):
        result = _sparkline([0.8])
        assert len(result) == 1


class TestGenerateTrendHtml:
    def test_generates_html(self, tmp_path: Path):
        output = tmp_path / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content

    def test_contains_svg(self, tmp_path: Path):
        output = tmp_path / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        content = output.read_text()
        assert "<svg" in content
        assert "</svg>" in content

    def test_contains_chart_titles(self, tmp_path: Path):
        output = tmp_path / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        content = output.read_text()
        assert "Average Score" in content
        assert "Pass Rate" in content
        assert "Total Cost" in content

    def test_contains_per_metric_chart(self, tmp_path: Path):
        output = tmp_path / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        content = output.read_text()
        assert "hallucination" in content

    def test_run_table(self, tmp_path: Path):
        output = tmp_path / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        content = output.read_text()
        assert "run-0" in content
        assert "v1" in content

    def test_empty_data(self, tmp_path: Path):
        output = tmp_path / "empty_trend.html"
        generate_trend_html([], output)
        assert output.exists()
        content = output.read_text()
        assert "No trend data" in content

    def test_single_run(self, tmp_path: Path):
        output = tmp_path / "single.html"
        generate_trend_html(_make_trend_data(1), output)
        assert output.exists()
        content = output.read_text()
        assert "<svg" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "sub" / "dir" / "trend.html"
        generate_trend_html(_make_trend_data(), output)
        assert output.exists()


class TestRenderTrendTerminal:
    def test_renders_to_string(self):
        output = render_trend_terminal(_make_trend_data(), to_string=True)
        assert output is not None
        assert "Trend" in output

    def test_contains_sparkline_rows(self):
        output = render_trend_terminal(_make_trend_data(), to_string=True)
        assert "Avg Score" in output
        assert "Pass Rate" in output
        assert "Cost" in output

    def test_contains_metric_rows(self):
        output = render_trend_terminal(_make_trend_data(), to_string=True)
        assert "hallucination" in output

    def test_contains_run_labels(self):
        output = render_trend_terminal(_make_trend_data(), to_string=True)
        assert "v1" in output
        assert "v3" in output

    def test_empty_data(self):
        output = render_trend_terminal([], to_string=True)
        assert output is not None
        assert "No trend data" in output

    def test_returns_none_without_to_string(self):
        result = render_trend_terminal(_make_trend_data(), to_string=False)
        assert result is None
