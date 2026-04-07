from pathlib import Path

import pytest

from checkllm.regression.snapshot import (
    Snapshot,
    TestRunRecord,
    MetricRecord,
    save_snapshot,
    load_snapshot,
)


class TestSnapshotModel:
    def test_create_snapshot(self):
        snap = Snapshot(
            version=1,
            tests={
                "test_foo": [
                    TestRunRecord(
                        metrics={
                            "hallucination": MetricRecord(score=0.9, passed=True)
                        }
                    )
                ]
            },
        )
        assert snap.version == 1
        assert "test_foo" in snap.tests
        assert snap.tests["test_foo"][0].metrics["hallucination"].score == 0.9

    def test_get_scores_for_metric(self):
        snap = Snapshot(
            version=1,
            tests={
                "test_foo": [
                    TestRunRecord(metrics={"h": MetricRecord(score=0.9, passed=True)}),
                    TestRunRecord(metrics={"h": MetricRecord(score=0.85, passed=True)}),
                    TestRunRecord(metrics={"h": MetricRecord(score=0.88, passed=True)}),
                ]
            },
        )
        scores = snap.get_scores("test_foo", "h")
        assert scores == [0.9, 0.85, 0.88]


class TestSaveAndLoadSnapshot:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        snap = Snapshot(
            version=1,
            tests={
                "test_bar": [
                    TestRunRecord(
                        metrics={"relevance": MetricRecord(score=0.8, passed=True)}
                    )
                ]
            },
        )
        filepath = tmp_path / "snapshot.json"
        save_snapshot(snap, filepath)
        loaded = load_snapshot(filepath)
        assert loaded.tests["test_bar"][0].metrics["relevance"].score == 0.8

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path / "nonexistent.json")

    def test_load_sample_fixture(self):
        path = Path(__file__).parent.parent / "fixtures" / "sample_snapshot.json"
        snap = load_snapshot(path)
        assert "test_summarizer_quality" in snap.tests
        scores = snap.get_scores("test_summarizer_quality", "hallucination")
        assert len(scores) == 2
