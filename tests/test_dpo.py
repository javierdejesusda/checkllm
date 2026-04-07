"""Tests for checkllm.dpo — DPO export and preference pair generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from checkllm.dpo import DPODataset, DPOExporter, DPOPair, DPOStats, ExportFormat


class TestDPOPair:
    def test_score_gap_computed(self):
        pair = DPOPair(
            prompt="hello",
            chosen="good",
            rejected="bad",
            chosen_score=0.9,
            rejected_score=0.3,
        )
        assert pair.score_gap == pytest.approx(0.6)

    def test_score_gap_when_rejected_higher(self):
        pair = DPOPair(
            prompt="hello",
            chosen="a",
            rejected="b",
            chosen_score=0.2,
            rejected_score=0.8,
        )
        assert pair.score_gap == pytest.approx(0.6)

    def test_score_gap_zero(self):
        pair = DPOPair(
            prompt="p",
            chosen="a",
            rejected="b",
            chosen_score=0.5,
            rejected_score=0.5,
        )
        assert pair.score_gap == pytest.approx(0.0)

    def test_metadata_default_empty(self):
        pair = DPOPair(
            prompt="p",
            chosen="a",
            rejected="b",
            chosen_score=0.9,
            rejected_score=0.1,
        )
        assert pair.metadata == {}

    def test_metadata_preserved(self):
        pair = DPOPair(
            prompt="p",
            chosen="a",
            rejected="b",
            chosen_score=0.9,
            rejected_score=0.1,
            metadata={"source": "test"},
        )
        assert pair.metadata["source"] == "test"

    def test_invalid_score_rejected(self):
        with pytest.raises(Exception):
            DPOPair(
                prompt="p",
                chosen="a",
                rejected="b",
                chosen_score=1.5,
                rejected_score=0.1,
            )


class TestExportFormat:
    def test_enum_values(self):
        assert ExportFormat.JSON == "json"
        assert ExportFormat.JSONL == "jsonl"
        assert ExportFormat.HUGGINGFACE == "huggingface"
        assert ExportFormat.OPENAI == "openai"


class TestDPOExporter:
    def _make_pair(
        self,
        prompt: str = "What is Python?",
        chosen: str = "A programming language",
        rejected: str = "A snake",
        chosen_score: float = 0.9,
        rejected_score: float = 0.2,
    ) -> DPOPair:
        return DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )

    def test_add_pair(self):
        exporter = DPOExporter()
        pair = self._make_pair()
        exporter.add_pair(pair)
        dataset = exporter.build()
        assert len(dataset.pairs) == 1
        assert dataset.pairs[0].prompt == "What is Python?"

    def test_add_multiple_pairs(self):
        exporter = DPOExporter()
        exporter.add_pair(self._make_pair(prompt="q1"))
        exporter.add_pair(self._make_pair(prompt="q2"))
        dataset = exporter.build()
        assert len(dataset.pairs) == 2

    def test_add_from_comparisons_basic(self):
        exporter = DPOExporter()
        exporter.add_from_comparisons(
            prompt="What is Python?",
            responses=[
                {"output": "A language", "score": 0.95},
                {"output": "A snake", "score": 0.2},
            ],
            min_score_gap=0.2,
        )
        dataset = exporter.build()
        assert len(dataset.pairs) == 1
        assert dataset.pairs[0].chosen == "A language"
        assert dataset.pairs[0].rejected == "A snake"

    def test_add_from_comparisons_min_gap_filters(self):
        exporter = DPOExporter()
        exporter.add_from_comparisons(
            prompt="Q",
            responses=[
                {"output": "A", "score": 0.5},
                {"output": "B", "score": 0.45},
            ],
            min_score_gap=0.2,
        )
        dataset = exporter.build()
        assert len(dataset.pairs) == 0

    def test_add_from_comparisons_three_responses(self):
        exporter = DPOExporter()
        exporter.add_from_comparisons(
            prompt="Q",
            responses=[
                {"output": "best", "score": 0.95},
                {"output": "mid", "score": 0.5},
                {"output": "worst", "score": 0.1},
            ],
            min_score_gap=0.2,
        )
        dataset = exporter.build()
        assert len(dataset.pairs) == 3
        gaps = sorted([p.score_gap for p in dataset.pairs], reverse=True)
        assert gaps[0] == pytest.approx(0.85)
        assert gaps[1] == pytest.approx(0.45)
        assert gaps[2] == pytest.approx(0.4)

    def test_add_from_comparisons_sorted_largest_gap_first(self):
        exporter = DPOExporter()
        exporter.add_from_comparisons(
            prompt="Q",
            responses=[
                {"output": "A", "score": 0.9},
                {"output": "B", "score": 0.6},
                {"output": "C", "score": 0.1},
            ],
            min_score_gap=0.0,
        )
        dataset = exporter.build()
        assert dataset.pairs[0].score_gap >= dataset.pairs[1].score_gap

    def test_add_from_comparisons_no_min_gap(self):
        exporter = DPOExporter()
        exporter.add_from_comparisons(
            prompt="Q",
            responses=[
                {"output": "A", "score": 0.51},
                {"output": "B", "score": 0.50},
            ],
            min_score_gap=0.0,
        )
        dataset = exporter.build()
        assert len(dataset.pairs) == 1

    def test_add_from_arena(self):
        exporter = DPOExporter()
        arena_results = [
            {
                "prompt": "What is AI?",
                "contestants": [
                    {"output": "response_best", "score": 0.9},
                    {"output": "response_mid", "score": 0.6},
                    {"output": "response_worst", "score": 0.2},
                ],
            },
        ]
        exporter.add_from_arena(arena_results)
        dataset = exporter.build()
        assert len(dataset.pairs) == 1
        assert dataset.pairs[0].chosen == "response_best"
        assert dataset.pairs[0].rejected == "response_worst"

    def test_add_from_arena_skips_ties(self):
        exporter = DPOExporter()
        arena_results = [
            {
                "prompt": "Q",
                "contestants": [
                    {"output": "A", "score": 0.5},
                    {"output": "B", "score": 0.5},
                ],
            },
        ]
        exporter.add_from_arena(arena_results)
        dataset = exporter.build()
        assert len(dataset.pairs) == 0

    def test_add_from_arena_skips_single_contestant(self):
        exporter = DPOExporter()
        arena_results = [
            {
                "prompt": "Q",
                "contestants": [{"output": "A", "score": 0.9}],
            },
        ]
        exporter.add_from_arena(arena_results)
        dataset = exporter.build()
        assert len(dataset.pairs) == 0

    def test_build_returns_dataset(self):
        exporter = DPOExporter()
        exporter.add_pair(self._make_pair())
        dataset = exporter.build()
        assert isinstance(dataset, DPODataset)
        assert dataset.source == "checkllm"
        assert dataset.created_at  # non-empty

    def test_build_does_not_share_reference(self):
        exporter = DPOExporter()
        exporter.add_pair(self._make_pair())
        ds1 = exporter.build()
        exporter.add_pair(self._make_pair(prompt="new"))
        ds2 = exporter.build()
        assert len(ds1.pairs) == 1
        assert len(ds2.pairs) == 2

    def test_clear(self):
        exporter = DPOExporter()
        exporter.add_pair(self._make_pair())
        exporter.clear()
        dataset = exporter.build()
        assert len(dataset.pairs) == 0


class TestDPODataset:
    def _make_dataset(self, n: int = 3) -> DPODataset:
        pairs = []
        for i in range(n):
            chosen_score = max(0.5, 0.9 - i * 0.01)
            rejected_score = min(0.4, 0.1 + i * 0.005)
            pairs.append(
                DPOPair(
                    prompt=f"question_{i}",
                    chosen=f"good_{i}",
                    rejected=f"bad_{i}",
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                )
            )
        return DPODataset(pairs=pairs, created_at="2025-01-01T00:00:00Z")

    def test_to_json_format(self):
        dataset = self._make_dataset(2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.to_json(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            assert isinstance(data, list)
            assert len(data) == 2
            assert set(data[0].keys()) == {"prompt", "chosen", "rejected"}
            assert data[0]["prompt"] == "question_0"
            assert data[0]["chosen"] == "good_0"
            assert data[0]["rejected"] == "bad_0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_jsonl_format(self):
        dataset = self._make_dataset(2)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            dataset.to_jsonl(path)
            text = Path(path).read_text(encoding="utf-8")
            lines = [ln for ln in text.strip().split("\n") if ln]
            assert len(lines) == 2
            for line in lines:
                record = json.loads(line)
                assert "prompt" in record
                assert "chosen" in record
                assert "rejected" in record
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_huggingface_format(self):
        dataset = self._make_dataset(2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.to_huggingface(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            assert isinstance(data, dict)
            assert set(data.keys()) == {"prompt", "chosen", "rejected"}
            assert isinstance(data["prompt"], list)
            assert len(data["prompt"]) == 2
            assert data["prompt"][0] == "question_0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_openai_format(self):
        dataset = self._make_dataset(1)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            dataset.to_openai(path)
            text = Path(path).read_text(encoding="utf-8")
            lines = [ln for ln in text.strip().split("\n") if ln]
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "messages" in record
            assert "weight" in record
            assert len(record["messages"]) == 2
            assert record["messages"][0]["role"] == "user"
            assert record["messages"][1]["role"] == "assistant"
            assert record["messages"][0]["content"] == "question_0"
            assert record["messages"][1]["content"] == "good_0"
            assert record["weight"] > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stats_basic(self):
        dataset = self._make_dataset(3)
        s = dataset.stats()
        assert isinstance(s, DPOStats)
        assert s.total_pairs == 3
        assert s.unique_prompts == 3
        assert s.avg_score_gap > 0
        assert s.max_score_gap >= s.min_score_gap
        assert s.avg_chosen_score > s.avg_rejected_score

    def test_stats_empty(self):
        dataset = DPODataset(pairs=[], created_at="2025-01-01T00:00:00Z")
        s = dataset.stats()
        assert s.total_pairs == 0
        assert s.avg_score_gap == 0.0
        assert s.unique_prompts == 0

    def test_filter_by_min_gap(self):
        pairs = [
            DPOPair(
                prompt="q1",
                chosen="a",
                rejected="b",
                chosen_score=0.9,
                rejected_score=0.1,
            ),
            DPOPair(
                prompt="q2",
                chosen="c",
                rejected="d",
                chosen_score=0.55,
                rejected_score=0.5,
            ),
        ]
        dataset = DPODataset(pairs=pairs, created_at="2025-01-01T00:00:00Z")
        filtered = dataset.filter(min_score_gap=0.3)
        assert len(filtered.pairs) == 1
        assert filtered.pairs[0].prompt == "q1"

    def test_filter_preserves_metadata(self):
        dataset = self._make_dataset(2)
        filtered = dataset.filter(min_score_gap=0.0)
        assert filtered.source == dataset.source
        assert filtered.created_at == dataset.created_at

    def test_sample_with_seed_reproducibility(self):
        dataset = self._make_dataset(10)
        sample1 = dataset.sample(n=3, seed=42)
        sample2 = dataset.sample(n=3, seed=42)
        assert len(sample1.pairs) == 3
        assert [p.prompt for p in sample1.pairs] == [p.prompt for p in sample2.pairs]

    def test_sample_different_seeds(self):
        dataset = self._make_dataset(20)
        sample1 = dataset.sample(n=5, seed=1)
        sample2 = dataset.sample(n=5, seed=2)
        prompts1 = [p.prompt for p in sample1.pairs]
        prompts2 = [p.prompt for p in sample2.pairs]
        assert prompts1 != prompts2

    def test_sample_clamped_to_size(self):
        dataset = self._make_dataset(3)
        sampled = dataset.sample(n=100)
        assert len(sampled.pairs) == 3
