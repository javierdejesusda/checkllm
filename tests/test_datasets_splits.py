"""Tests for reproducible train/test and k-fold splitting."""

from __future__ import annotations

import pytest

from checkllm.datasets.case import Case
from checkllm.datasets.splits import k_fold_split, train_test_split


def _make_cases(n: int, labels: list[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    for i in range(n):
        meta = {"label": labels[i]} if labels else {"idx": i}
        cases.append(Case(input=f"case-{i}", metadata=meta))
    return cases


def test_train_test_split_reproducible_with_same_seed():
    cases = _make_cases(50)
    train_a, test_a = train_test_split(cases, test_size=0.2, seed=123)
    train_b, test_b = train_test_split(cases, test_size=0.2, seed=123)
    assert [c.input for c in train_a] == [c.input for c in train_b]
    assert [c.input for c in test_a] == [c.input for c in test_b]


def test_train_test_split_differs_with_different_seeds():
    cases = _make_cases(50)
    _, test_a = train_test_split(cases, test_size=0.2, seed=1)
    _, test_b = train_test_split(cases, test_size=0.2, seed=2)
    assert [c.input for c in test_a] != [c.input for c in test_b]


def test_train_test_split_sizes():
    cases = _make_cases(100)
    train, test = train_test_split(cases, test_size=0.25, seed=0)
    assert len(train) + len(test) == 100
    assert len(test) == 25


def test_train_test_split_rejects_invalid_size():
    with pytest.raises(ValueError):
        train_test_split(_make_cases(10), test_size=0.0)
    with pytest.raises(ValueError):
        train_test_split(_make_cases(10), test_size=1.0)


def test_train_test_split_stratified_preserves_class_ratio():
    labels = ["A"] * 80 + ["B"] * 20
    cases = _make_cases(100, labels=labels)
    train, test = train_test_split(cases, test_size=0.2, seed=42, stratify_by="metadata.label")
    train_b = sum(1 for c in train if c.metadata["label"] == "B")
    test_b = sum(1 for c in test if c.metadata["label"] == "B")
    assert train_b + test_b == 20
    # Stratified split must place some of each class in each side.
    assert test_b >= 1
    assert train_b >= 1


def test_train_test_split_empty_input():
    train, test = train_test_split([], test_size=0.2)
    assert train == []
    assert test == []


def test_k_fold_split_covers_all_cases_exactly_once():
    cases = _make_cases(20)
    folds = k_fold_split(cases, k=5, seed=7)
    assert len(folds) == 5
    seen: set[str] = set()
    for _, test in folds:
        for c in test:
            assert c.input not in seen
            seen.add(c.input)
    assert seen == {c.input for c in cases}


def test_k_fold_split_reproducible():
    cases = _make_cases(15)
    f1 = k_fold_split(cases, k=3, seed=9)
    f2 = k_fold_split(cases, k=3, seed=9)
    for (_, t1), (_, t2) in zip(f1, f2):
        assert [c.input for c in t1] == [c.input for c in t2]


def test_k_fold_split_rejects_invalid_k():
    with pytest.raises(ValueError):
        k_fold_split(_make_cases(10), k=1)
    with pytest.raises(ValueError):
        k_fold_split(_make_cases(3), k=5)
