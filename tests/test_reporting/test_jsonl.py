"""Tests for JSONL export."""
import json


from checkllm.models import CheckResult
from checkllm.reporting.jsonl import export_jsonl


def _sample_results():
    return {
        "test_foo": [
            CheckResult(passed=True, score=0.9, reasoning="Good", cost=0.001, latency_ms=100, metric_name="hallucination"),
        ],
        "test_bar": [
            CheckResult(passed=False, score=0.3, reasoning="Bad", cost=0.002, latency_ms=200, metric_name="relevance"),
            CheckResult(passed=True, score=1.0, reasoning="OK", cost=0.0, latency_ms=0, metric_name="contains"),
        ],
    }


class TestJsonlExport:
    def test_generates_string(self):
        text = export_jsonl(_sample_results())
        lines = text.strip().split("\n")
        assert len(lines) == 3

    def test_each_line_is_valid_json(self):
        text = export_jsonl(_sample_results())
        for line in text.strip().split("\n"):
            record = json.loads(line)
            assert "test" in record
            assert "metric" in record
            assert "score" in record
            assert "passed" in record

    def test_preserves_data(self):
        text = export_jsonl(_sample_results())
        records = [json.loads(line) for line in text.strip().split("\n")]
        assert records[0]["test"] == "test_foo"
        assert records[0]["metric"] == "hallucination"
        assert records[0]["score"] == 0.9
        assert records[1]["passed"] is False

    def test_writes_to_file(self, tmp_path):
        output = tmp_path / "results.jsonl"
        export_jsonl(_sample_results(), output)
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_empty_results(self):
        text = export_jsonl({})
        assert text == ""
