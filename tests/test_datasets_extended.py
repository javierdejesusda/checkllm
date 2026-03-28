"""Tests for CSV and JSON dataset loading."""
import json
from pathlib import Path

import pytest

from checkllm.datasets.loader import load_csv_dataset, load_json_dataset, load_dataset


class TestJsonDataset:
    def test_load_json_array(self, tmp_path):
        data = [
            {"input": "What is Python?", "expected": "A programming language"},
            {"input": "What is 2+2?", "expected": "4", "query": "math"},
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))

        cases = load_json_dataset(path)
        assert len(cases) == 2
        assert cases[0].input == "What is Python?"
        assert cases[0].expected == "A programming language"
        assert cases[1].query == "math"

    def test_load_json_with_metadata(self, tmp_path):
        data = [{"input": "test", "context": "ctx", "criteria": "be good"}]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))

        cases = load_json_dataset(path)
        assert cases[0].context == "ctx"
        assert cases[0].criteria == "be good"

    def test_load_json_not_array_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_json_dataset(path)

    def test_load_json_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json_dataset(tmp_path / "missing.json")


class TestCsvDataset:
    def test_load_csv(self, tmp_path):
        csv_content = "input,expected,query\nWhat is Python?,A language,explain\nWhat is 2+2?,4,math\n"
        path = tmp_path / "cases.csv"
        path.write_text(csv_content)

        cases = load_csv_dataset(path)
        assert len(cases) == 2
        assert cases[0].input == "What is Python?"
        assert cases[0].expected == "A language"
        assert cases[1].query == "math"

    def test_load_csv_with_extra_columns(self, tmp_path):
        csv_content = "input,expected,difficulty,category\nhello,world,easy,test\n"
        path = tmp_path / "cases.csv"
        path.write_text(csv_content)

        cases = load_csv_dataset(path)
        assert len(cases) == 1
        assert cases[0].input == "hello"
        assert cases[0].metadata["difficulty"] == "easy"
        assert cases[0].metadata["category"] == "test"

    def test_load_csv_empty_optional_fields(self, tmp_path):
        csv_content = "input,expected,query\nhello,,\n"
        path = tmp_path / "cases.csv"
        path.write_text(csv_content)

        cases = load_csv_dataset(path)
        assert cases[0].input == "hello"
        assert cases[0].expected is None
        assert cases[0].query is None

    def test_load_csv_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv_dataset(tmp_path / "missing.csv")


class TestLoadDatasetAutoDetect:
    def test_json_autodetect(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([{"input": "test"}]))
        cases = load_dataset(path)
        assert len(cases) == 1

    def test_csv_autodetect(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_text("input,expected\nhello,world\n")
        cases = load_dataset(path)
        assert len(cases) == 1

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "cases.parquet"
        path.write_text("whatever")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(path)
