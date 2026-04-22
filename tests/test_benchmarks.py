"""Tests for the checkllm.benchmarks subpackage."""

from __future__ import annotations

import pytest

from checkllm.benchmarks.datasets import (
    BenchmarkDataset,
    BenchmarkSample,
    list_benchmarks,
    load_benchmark,
)
from checkllm.benchmarks.runner import BenchmarkResult, BenchmarkSuite


def test_list_benchmarks_returns_expected_names():
    names = list_benchmarks()
    expected = {
        "arc",
        "bbh",
        "bbq",
        "boolq",
        "drop",
        "gsm8k",
        "hellaswag",
        "humaneval",
        "ifeval",
        "lambada",
        "logiqa",
        "mathqa",
        "mmlu",
        "squad",
        "truthfulqa",
        "winogrande",
    }
    assert set(names) == expected


def test_list_benchmarks_is_sorted():
    names = list_benchmarks()
    assert names == sorted(names)


def test_load_mmlu_limit():
    dataset = load_benchmark("mmlu", limit=5)
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.name == "mmlu"
    assert len(dataset.samples) == 5
    for sample in dataset.samples:
        assert isinstance(sample, BenchmarkSample)
        assert sample.question
        assert sample.correct_answer


def test_load_mmlu_has_choices():
    dataset = load_benchmark("mmlu", limit=5)
    for sample in dataset.samples:
        assert sample.choices is not None
        assert len(sample.choices) == 4


def test_load_mmlu_spans_multiple_categories():
    dataset = load_benchmark("mmlu")
    categories = {s.category for s in dataset.samples}
    assert len(categories) >= 5


def test_load_truthfulqa_limit():
    dataset = load_benchmark("truthfulqa", limit=5)
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.name == "truthfulqa"
    assert len(dataset.samples) == 5
    for sample in dataset.samples:
        assert sample.question
        assert sample.correct_answer


def test_load_gsm8k_limit():
    dataset = load_benchmark("gsm8k", limit=5)
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.name == "gsm8k"
    assert len(dataset.samples) == 5
    for sample in dataset.samples:
        assert sample.question
        assert sample.correct_answer


def test_load_benchmark_unknown_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("does_not_exist")


def test_load_benchmark_case_insensitive():
    dataset = load_benchmark("MMLU", limit=3)
    assert dataset.name == "mmlu"
    assert len(dataset.samples) == 3


def test_all_benchmarks_have_at_least_15_samples():
    for name in list_benchmarks():
        dataset = load_benchmark(name)
        assert len(dataset.samples) >= 15, f"{name} has only {len(dataset.samples)} samples"


def test_benchmark_result_summary_contains_name_and_accuracy():
    result = BenchmarkResult(
        benchmark="mmlu",
        total=10,
        correct=7,
        accuracy=0.7,
        by_category={"math": 0.5, "science": 1.0},
    )
    summary = result.summary()
    assert "mmlu" in summary
    assert "70.0%" in summary


def test_benchmark_result_summary_contains_categories():
    result = BenchmarkResult(
        benchmark="gsm8k",
        total=5,
        correct=4,
        accuracy=0.8,
        by_category={"arithmetic": 1.0},
    )
    summary = result.summary()
    assert "arithmetic" in summary


def test_benchmark_result_zero_total():
    result = BenchmarkResult(
        benchmark="truthfulqa",
        total=0,
        correct=0,
        accuracy=0.0,
    )
    assert result.accuracy == 0.0
    assert "truthfulqa" in result.summary()


def test_benchmark_suite_construction():
    suite = BenchmarkSuite(benchmarks=["mmlu", "gsm8k"])
    assert "mmlu" in suite.benchmarks
    assert "gsm8k" in suite.benchmarks
    assert suite.limit_per_benchmark == 100


def test_benchmark_suite_custom_limit():
    suite = BenchmarkSuite(benchmarks=["mmlu"], limit_per_benchmark=25)
    assert suite.limit_per_benchmark == 25


def test_benchmark_sample_optional_fields():
    sample = BenchmarkSample(
        question="What is 2 + 2?",
        correct_answer="4",
    )
    assert sample.choices is None
    assert sample.category is None


def test_load_benchmark_no_limit_returns_all():
    dataset = load_benchmark("gsm8k")
    assert len(dataset.samples) == 20
