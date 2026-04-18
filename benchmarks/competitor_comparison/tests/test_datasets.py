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


def test_ragtruth_parses_json_string_hallucination_labels():
    rows = [
        {
            "id": "rt-empty",
            "query": "q",
            "context": "c",
            "output": "o",
            "hallucination_labels": "[]",
        },
        {
            "id": "rt-hall",
            "query": "q",
            "context": "c",
            "output": "o",
            "hallucination_labels": '[{"type": "evident_conflict"}]',
        },
    ]
    samples = load_ragtruth_from_rows(rows)
    labels = {s.sample_id: s.ground_truth.label for s in samples}
    assert labels["rt-empty"] == 1.0
    assert labels["rt-hall"] == 0.0


def test_truthfulqa_emits_balanced_correct_and_incorrect_samples(tiny_truthfulqa):
    """Each source row expands into a correct (label=1.0) and incorrect
    (label=0.0) sample pair so the slice has two classes for ROC-AUC.
    """
    samples = load_truthfulqa_from_rows(tiny_truthfulqa)
    assert len(samples) == 6
    positives = [s for s in samples if s.ground_truth.label == 1.0]
    negatives = [s for s in samples if s.ground_truth.label == 0.0]
    assert len(positives) == 3
    assert len(negatives) == 3
    first_correct = next(s for s in samples if s.sample_id == "tq-1-correct")
    first_incorrect = next(s for s in samples if s.sample_id == "tq-1-incorrect")
    assert first_correct.query.startswith("What happens if you eat watermelon")
    assert first_correct.answer == "Nothing harmful; they pass through digestion."
    assert first_correct.context == first_correct.answer
    assert first_incorrect.answer == "A watermelon will grow in your stomach."
    assert first_incorrect.context == "Nothing harmful; they pass through digestion."
    assert first_incorrect.query == first_correct.query


def test_truthfulqa_skips_rows_without_incorrect_answers():
    """A row without any incorrect answers cannot contribute a negative
    sample, so it is dropped entirely rather than emitting an unusable
    label=1.0 singleton.
    """
    rows = [
        {
            "question": "q0",
            "best_answer": "a0",
            "incorrect_answers": ["bad0"],
        },
        {"question": "q1", "best_answer": "a1", "incorrect_answers": []},
        {"question": "q2", "best_answer": "a2"},
    ]
    samples = load_truthfulqa_from_rows(rows)
    assert len(samples) == 2
    assert {s.sample_id for s in samples} == {"tq-0-correct", "tq-0-incorrect"}


def test_truthfulqa_preserves_explicit_zero_id():
    """A row that ships id=0 must be preserved, not silently rewritten to
    the positional fallback. The previous `row.get("id") or f"tq-{idx}"`
    mapping would discard 0, "", and False.
    """
    rows = [
        {"id": 0, "question": "q0", "best_answer": "a0", "incorrect_answers": ["b0"]},
        {"question": "q1", "best_answer": "a1", "incorrect_answers": ["b1"]},
    ]
    samples = load_truthfulqa_from_rows(rows)
    ids = [s.sample_id for s in samples]
    assert ids == ["0-correct", "0-incorrect", "tq-1-correct", "tq-1-incorrect"]


def test_jailbreakbench_harmful_is_zero(tiny_jailbreakbench):
    samples = load_jailbreakbench_from_rows(tiny_jailbreakbench)
    assert len(samples) == 3
    harmful = [s for s in samples if s.ground_truth.label == 0.0]
    benign = [s for s in samples if s.ground_truth.label == 1.0]
    assert len(harmful) == 2
    assert len(benign) == 1
