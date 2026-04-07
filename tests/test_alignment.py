from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from checkllm.alignment import (
    AlignmentResult,
    HumanLabel,
    MetricAligner,
    _pearson_correlation,
)
from checkllm.models import JudgeResponse


def _make_label(
    human_score: float, output: str = "test output", query: str = "test query"
) -> HumanLabel:
    """Create a HumanLabel with the given score and sensible defaults."""
    return HumanLabel(
        output=output,
        context="test context",
        query=query,
        human_score=human_score,
    )


def _make_labels(scores: list[float]) -> list[HumanLabel]:
    """Create a list of HumanLabels with varying outputs."""
    return [
        _make_label(s, output=f"output_{i}", query=f"query_{i}")
        for i, s in enumerate(scores)
    ]


class TestHumanLabel:
    def test_valid_label(self):
        label = HumanLabel(
            output="The sky is blue",
            context="Scientific facts about the sky",
            query="What color is the sky?",
            human_score=0.9,
        )
        assert label.human_score == 0.9
        assert label.reference is None
        assert label.metadata is None

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValueError):
            HumanLabel(
                output="x", context="c", query="q", human_score=-0.1
            )

    def test_score_above_one_rejected(self):
        with pytest.raises(ValueError):
            HumanLabel(
                output="x", context="c", query="q", human_score=1.1
            )

    def test_optional_fields(self):
        label = HumanLabel(
            output="x",
            context="c",
            query="q",
            human_score=0.5,
            reference="gold answer",
            metadata={"source": "test"},
        )
        assert label.reference == "gold answer"
        assert label.metadata == {"source": "test"}


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1, 2, 3, 4], [10, 20, 30, 40])
        assert abs(r - 1.0) < 1e-10

    def test_perfect_negative(self):
        r = _pearson_correlation([1, 2, 3, 4], [40, 30, 20, 10])
        assert abs(r - (-1.0)) < 1e-10

    def test_no_correlation(self):
        r = _pearson_correlation([1, 2, 3, 4], [1, 1, 1, 1])
        assert r == 0.0

    def test_single_element(self):
        r = _pearson_correlation([1.0], [1.0])
        assert r == 0.0

    def test_empty(self):
        r = _pearson_correlation([], [])
        assert r == 0.0


class TestSelectFewShot:
    def test_selects_diverse_scores(self):
        labels = _make_labels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        aligner = MetricAligner(judge=AsyncMock(), seed=42)
        selected = aligner._select_few_shot(labels, n=3)

        assert len(selected) == 3
        scores = sorted(lb.human_score for lb in selected)
        assert scores[0] == 0.0
        assert scores[-1] == 1.0

    def test_returns_all_if_fewer_than_n(self):
        labels = _make_labels([0.3, 0.7])
        aligner = MetricAligner(judge=AsyncMock(), seed=42)
        selected = aligner._select_few_shot(labels, n=5)
        assert len(selected) == 2

    def test_single_example(self):
        labels = _make_labels([0.1, 0.5, 0.9])
        aligner = MetricAligner(judge=AsyncMock(), seed=42)
        selected = aligner._select_few_shot(labels, n=1)
        assert len(selected) == 1


class TestEvaluateCorrelation:
    @pytest.mark.asyncio
    async def test_computes_correlation_with_mock_judge(self):
        mock_judge = AsyncMock()
        labels = _make_labels([0.2, 0.5, 0.8])

        call_count = 0

        async def fake_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            scores = [0.25, 0.55, 0.75]
            score = scores[call_count % len(scores)]
            call_count += 1
            return JudgeResponse(score=score, reasoning="ok", cost=0.001)

        mock_judge.evaluate = AsyncMock(side_effect=fake_evaluate)

        aligner = MetricAligner(judge=mock_judge, seed=42)
        r = await aligner._evaluate_correlation("test prompt", labels)

        assert mock_judge.evaluate.call_count == 3
        assert r > 0.9


class TestAlign:
    @pytest.mark.asyncio
    async def test_align_improves_or_maintains_correlation(self):
        """Verify align() runs without error and returns valid result."""
        iteration_count = 0

        async def fake_evaluate(prompt, system_prompt=None):
            nonlocal iteration_count
            iteration_count += 1
            score = min(1.0, 0.3 + iteration_count * 0.02)
            raw = json.dumps({"prompt": "improved prompt", "score": score})
            return JudgeResponse(
                score=score, reasoning="evaluated", raw_output=raw, cost=0.001
            )

        mock_judge = AsyncMock()
        mock_judge.evaluate = AsyncMock(side_effect=fake_evaluate)

        labels = _make_labels([0.1, 0.3, 0.5, 0.7, 0.9])
        aligner = MetricAligner(judge=mock_judge, seed=42)
        result = await aligner.align(
            metric_name="faithfulness",
            labels=labels,
            iterations=2,
            strategy="both",
        )

        assert isinstance(result, AlignmentResult)
        assert result.metric_name == "faithfulness"
        assert result.iterations_run == 2
        assert result.aligned_prompt is not None
        assert result.original_prompt is not None

    @pytest.mark.asyncio
    async def test_align_rejects_too_few_labels(self):
        mock_judge = AsyncMock()
        aligner = MetricAligner(judge=mock_judge)
        with pytest.raises(ValueError, match="At least 2 labels"):
            await aligner.align("test", [_make_label(0.5)], iterations=1)

    @pytest.mark.asyncio
    async def test_align_rejects_invalid_strategy(self):
        mock_judge = AsyncMock()
        aligner = MetricAligner(judge=mock_judge)
        labels = _make_labels([0.3, 0.7])
        with pytest.raises(ValueError, match="strategy"):
            await aligner.align("test", labels, strategy="invalid")


class TestApply:
    def test_apply_sets_system_prompt(self):
        mock_judge = AsyncMock()
        aligner = MetricAligner(judge=mock_judge)

        class FakeMetric:
            system_prompt = "original"

        metric = FakeMetric()
        result = AlignmentResult(
            metric_name="test",
            aligned_prompt="aligned prompt text",
            original_prompt="original",
            correlation_before=0.5,
            correlation_after=0.8,
            improvement=60.0,
            iterations_run=3,
        )
        aligner.apply(metric, result)
        assert metric.system_prompt == "aligned prompt text"

    def test_apply_raises_if_no_system_prompt(self):
        mock_judge = AsyncMock()
        aligner = MetricAligner(judge=mock_judge)

        result = AlignmentResult(
            metric_name="test",
            aligned_prompt="new",
            original_prompt="old",
            correlation_before=0.5,
            correlation_after=0.8,
            improvement=60.0,
            iterations_run=1,
        )
        with pytest.raises(AttributeError, match="system_prompt"):
            aligner.apply(object(), result)


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        mock_judge = AsyncMock()
        aligner = MetricAligner(judge=mock_judge)

        result = AlignmentResult(
            metric_name="faithfulness",
            aligned_prompt="aligned",
            original_prompt="original",
            correlation_before=0.4,
            correlation_after=0.85,
            improvement=112.5,
            few_shot_examples=[{"output": "x", "query": "q", "human_score": 0.9}],
            iterations_run=5,
        )

        path = tmp_path / "alignment.json"
        aligner.save(result, path)
        assert path.exists()

        loaded = aligner.load(path)
        assert loaded.metric_name == result.metric_name
        assert loaded.aligned_prompt == result.aligned_prompt
        assert loaded.correlation_before == result.correlation_before
        assert loaded.correlation_after == result.correlation_after
        assert loaded.improvement == result.improvement
        assert loaded.few_shot_examples == result.few_shot_examples
        assert loaded.iterations_run == result.iterations_run
