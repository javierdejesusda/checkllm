"""Tests for CSV export module."""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path


from checkllm.models import CheckResult
from checkllm.reporting.csv_export import results_to_dataframe, write_csv, write_csv_string


def _sample_results() -> dict[str, list[CheckResult]]:
    return {
        "test_foo": [
            CheckResult(
                passed=True, score=0.9, reasoning="Good",
                cost=0.001, latency_ms=100, metric_name="hallucination",
            ),
        ],
        "test_bar": [
            CheckResult(
                passed=False, score=0.3, reasoning="Bad",
                cost=0.002, latency_ms=200, metric_name="relevance",
            ),
            CheckResult(
                passed=True, score=1.0, reasoning="OK",
                cost=0.0, latency_ms=0, metric_name="contains",
            ),
        ],
    }


class TestResultsToDataframe:
    def test_returns_list_of_dicts(self):
        rows = results_to_dataframe(_sample_results())
        assert isinstance(rows, list)
        assert len(rows) == 3
        assert all(isinstance(r, dict) for r in rows)

    def test_dict_keys(self):
        rows = results_to_dataframe(_sample_results())
        expected_keys = {"test_name", "metric_name", "passed", "score", "reasoning", "cost", "latency_ms"}
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_preserves_data(self):
        rows = results_to_dataframe(_sample_results())
        assert rows[0]["test_name"] == "test_foo"
        assert rows[0]["metric_name"] == "hallucination"
        assert rows[0]["score"] == 0.9
        assert rows[0]["passed"] is True
        assert rows[1]["passed"] is False

    def test_empty_results(self):
        rows = results_to_dataframe({})
        assert rows == []


class TestWriteCsv:
    def test_writes_file(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        write_csv(_sample_results(), output)
        assert output.exists()

    def test_csv_has_header_and_data(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        write_csv(_sample_results(), output)
        reader = csv.DictReader(output.open(newline="", encoding="utf-8"))
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["test_name"] == "test_foo"
        assert rows[0]["metric_name"] == "hallucination"
        assert rows[0]["score"] == "0.9"
        assert rows[0]["passed"] == "True"

    def test_csv_columns(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        write_csv(_sample_results(), output)
        reader = csv.DictReader(output.open(newline="", encoding="utf-8"))
        assert reader.fieldnames == [
            "test_name", "metric_name", "passed", "score",
            "reasoning", "cost", "latency_ms",
        ]

    def test_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "sub" / "dir" / "results.csv"
        write_csv(_sample_results(), output)
        assert output.exists()

    def test_empty_results_writes_header_only(self, tmp_path: Path):
        output = tmp_path / "empty.csv"
        write_csv({}, output)
        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only


class TestWriteCsvString:
    def test_returns_string(self):
        text = write_csv_string(_sample_results())
        assert isinstance(text, str)
        assert "test_name" in text  # header present

    def test_round_trips_with_csv_reader(self):
        text = write_csv_string(_sample_results())
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["test_name"] == "test_foo"
