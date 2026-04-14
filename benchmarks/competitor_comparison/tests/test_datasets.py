from bench.datasets import (
    load_halubench_from_rows,
    load_jailbreakbench_from_rows,
    load_ragtruth_from_rows,
    load_truthfulqa_from_rows,
)
from bench.schema import BenchmarkSample


def test_halubench_maps_pass_to_one(tiny_halubench):
    samples = load_halubench_from_rows(tiny_halubench)
    assert len(samples) == 3
    pass_samples = [s for s in samples if s.ground_truth.label == 1.0]
    fail_samples = [s for s in samples if s.ground_truth.label == 0.0]
    assert len(pass_samples) == 2
    assert len(fail_samples) == 1
    assert all(isinstance(s, BenchmarkSample) for s in samples)
    assert all(s.dataset == "halubench" for s in samples)


def test_ragtruth_any_label_means_hallucinated(tiny_ragtruth):
    samples = load_ragtruth_from_rows(tiny_ragtruth)
    assert len(samples) == 3
    faithful = [s for s in samples if s.ground_truth.label == 1.0]
    hallucinated = [s for s in samples if s.ground_truth.label == 0.0]
    assert len(faithful) == 1
    assert len(hallucinated) == 2


def test_truthfulqa_extracts_best_answer_as_reference(tiny_truthfulqa):
    samples = load_truthfulqa_from_rows(tiny_truthfulqa)
    assert len(samples) == 3
    assert samples[0].query.startswith("What happens if you eat watermelon")
    assert "Nothing harmful" in samples[0].context


def test_jailbreakbench_harmful_is_zero(tiny_jailbreakbench):
    samples = load_jailbreakbench_from_rows(tiny_jailbreakbench)
    assert len(samples) == 3
    harmful = [s for s in samples if s.ground_truth.label == 0.0]
    benign = [s for s in samples if s.ground_truth.label == 1.0]
    assert len(harmful) == 2
    assert len(benign) == 1
