"""Tests for the new HuggingFace-derived benchmarks.

Covers SQuAD 2.0 (``squad_v2``), ARC-Challenge (``arc_challenge``), BIG-Bench
Hard (``bbh_hard``), DROP-Reading (``drop_reading``), and CNN/DailyMail
(``cnn_dailymail``). Tiny fixture subsets are shipped in
``checkllm.benchmarks.datasets``, so these tests do not hit the network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from checkllm.benchmarks.datasets import load_benchmark
from checkllm.benchmarks.runner import BenchmarkRunner


NEW_BENCHMARKS = [
    "squad_v2",
    "arc_challenge",
    "bbh_hard",
    "drop_reading",
    "cnn_dailymail",
]


@pytest.mark.parametrize("name", NEW_BENCHMARKS)
def test_new_benchmark_has_minimum_fixture_samples(name: str) -> None:
    dataset = load_benchmark(name)
    assert len(dataset.samples) >= 15


@pytest.mark.parametrize("name", NEW_BENCHMARKS)
def test_new_benchmark_samples_have_required_fields(name: str) -> None:
    dataset = load_benchmark(name)
    for sample in dataset.samples:
        assert sample.question
        assert sample.correct_answer
        assert sample.category


def test_squad_v2_has_answerable_and_unanswerable() -> None:
    dataset = load_benchmark("squad_v2")
    categories = {sample.category for sample in dataset.samples}
    assert "answerable" in categories
    assert "unanswerable" in categories


def test_arc_challenge_samples_are_multiple_choice() -> None:
    dataset = load_benchmark("arc_challenge")
    for sample in dataset.samples:
        assert sample.choices is not None
        assert len(sample.choices) == 4


def test_bbh_hard_samples_are_multiple_choice() -> None:
    dataset = load_benchmark("bbh_hard")
    for sample in dataset.samples:
        assert sample.choices is not None
        assert len(sample.choices) >= 2


def test_drop_reading_samples_are_open_ended() -> None:
    dataset = load_benchmark("drop_reading")
    for sample in dataset.samples:
        assert sample.choices is None


def test_cnn_dailymail_samples_are_open_ended_summaries() -> None:
    dataset = load_benchmark("cnn_dailymail")
    for sample in dataset.samples:
        assert sample.choices is None
        assert "Summarize" in sample.question


def _mock_provider(answer_text: str) -> MagicMock:
    """Return a provider whose ``evaluate`` always replies with ``answer_text``."""
    response = MagicMock()
    response.cost = 0.0
    response.raw_output = answer_text
    response.reasoning = answer_text
    provider = MagicMock()
    provider.evaluate = AsyncMock(return_value=response)
    return provider


@pytest.mark.asyncio
async def test_arc_challenge_runner_scores_correct_letter() -> None:
    """If the mock always returns 'B', only samples with answer 'B' count."""
    provider = _mock_provider("B")
    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("arc_challenge")
    dataset = load_benchmark("arc_challenge")
    expected_correct = sum(1 for s in dataset.samples if s.correct_answer == "B")
    assert result.total == len(dataset.samples)
    assert result.correct == expected_correct
    assert result.accuracy == pytest.approx(expected_correct / result.total)


@pytest.mark.asyncio
async def test_bbh_hard_runner_scores_correct_letter() -> None:
    provider = _mock_provider("B")
    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("bbh_hard")
    dataset = load_benchmark("bbh_hard")
    expected_correct = sum(1 for s in dataset.samples if s.correct_answer == "B")
    assert result.correct == expected_correct


@pytest.mark.asyncio
async def test_squad_v2_runner_abstains_on_unanswerable() -> None:
    """A model that always replies 'unanswerable' scores all unanswerable items."""
    provider = _mock_provider("unanswerable")
    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("squad_v2")
    dataset = load_benchmark("squad_v2")
    expected_correct = sum(1 for s in dataset.samples if s.correct_answer == "unanswerable")
    assert result.correct == expected_correct


@pytest.mark.asyncio
async def test_squad_v2_runner_hits_answerable_on_match() -> None:
    """Pre-canned correct answers for each sample yield perfect accuracy."""
    dataset = load_benchmark("squad_v2", limit=3)

    canned = [s.correct_answer for s in dataset.samples]
    responses = [MagicMock(cost=0.0, raw_output=a, reasoning=a) for a in canned]
    provider = MagicMock()
    provider.evaluate = AsyncMock(side_effect=responses)

    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("squad_v2", limit=3)
    assert result.correct == 3
    assert result.accuracy == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_drop_reading_runner_numeric_accuracy() -> None:
    """Canned numeric answers for the first three samples should all match."""
    dataset = load_benchmark("drop_reading", limit=3)
    canned = [s.correct_answer for s in dataset.samples]
    responses = [MagicMock(cost=0.0, raw_output=a, reasoning=a) for a in canned]
    provider = MagicMock()
    provider.evaluate = AsyncMock(side_effect=responses)

    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("drop_reading", limit=3)
    assert result.correct == 3


@pytest.mark.asyncio
async def test_cnn_dailymail_runner_rewards_reference_summary() -> None:
    """When the model returns the gold summary, BLEU+ROUGE should pass."""
    dataset = load_benchmark("cnn_dailymail", limit=3)
    canned = [s.correct_answer for s in dataset.samples]
    responses = [MagicMock(cost=0.0, raw_output=a, reasoning=a) for a in canned]
    provider = MagicMock()
    provider.evaluate = AsyncMock(side_effect=responses)

    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("cnn_dailymail", limit=3)
    assert result.correct == 3
    assert result.accuracy == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_cnn_dailymail_runner_penalises_garbage_output() -> None:
    """A junk summary should score poorly."""
    provider = _mock_provider("zzz qqq xxx")
    runner = BenchmarkRunner(provider=provider)
    result = await runner.arun("cnn_dailymail", limit=5)
    assert result.accuracy < 0.5


def test_qa_token_f1_identical_is_one() -> None:
    assert BenchmarkRunner._qa_token_f1("Neil Armstrong", "Neil Armstrong") == pytest.approx(1.0)


def test_qa_token_f1_disjoint_is_zero() -> None:
    assert BenchmarkRunner._qa_token_f1("red apple", "blue sky") == 0.0


def test_qa_token_f1_partial_overlap() -> None:
    f1 = BenchmarkRunner._qa_token_f1("the quick brown fox", "the lazy brown dog")
    assert 0.0 < f1 < 1.0


def test_bleu_score_identical_is_high() -> None:
    text = "the cat sat on the mat"
    assert BenchmarkRunner._bleu_score(text, text) == pytest.approx(1.0)


def test_rouge_l_identical_is_one() -> None:
    text = "alpha beta gamma delta"
    assert BenchmarkRunner._rouge_l_f1(text, text) == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero() -> None:
    assert BenchmarkRunner._rouge_l_f1("one two three", "four five six") == 0.0


def test_check_squad_v2_unanswerable_detection() -> None:
    assert BenchmarkRunner._check_squad_v2("unanswerable", "unanswerable")
    assert BenchmarkRunner._check_squad_v2("No answer in passage", "unanswerable")
    assert not BenchmarkRunner._check_squad_v2("Paris", "unanswerable")


def test_check_squad_v2_rejects_abstain_on_answerable() -> None:
    assert not BenchmarkRunner._check_squad_v2("unanswerable", "Neil Armstrong")


def test_check_summary_passes_on_overlap() -> None:
    reference = "The central bank raised interest rates by 0.25 percentage points"
    close = "The central bank raised interest rates by a quarter point"
    assert BenchmarkRunner._check_summary(close, reference)


def test_check_summary_fails_on_garbage() -> None:
    assert not BenchmarkRunner._check_summary("zzz qqq", "Real summary text")
