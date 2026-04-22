"""Reproducible train/test and k-fold splitting for Case datasets.

Pure-Python implementation that intentionally avoids depending on scikit-learn.
All splits use ``random.Random(seed)`` so the same seed always yields the same
split, across runs and platforms.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable

from checkllm.datasets.case import Case


def _resolve_stratify_key(case: Case, stratify_by: str) -> Any:
    """Extract a stratification key from a Case.

    The key can reference a top-level Case attribute (``"expected"``) or a
    ``metadata.*`` entry.  Missing values are returned as ``None``.
    """
    if stratify_by.startswith("metadata."):
        meta_key = stratify_by.split(".", 1)[1]
        return case.metadata.get(meta_key)
    if hasattr(case, stratify_by):
        return getattr(case, stratify_by)
    return case.metadata.get(stratify_by)


def train_test_split(
    cases: list[Case],
    test_size: float = 0.2,
    seed: int = 42,
    stratify_by: str | None = None,
) -> tuple[list[Case], list[Case]]:
    """Split ``cases`` into reproducible train/test lists.

    Args:
        cases: The cases to split.
        test_size: Fraction of cases placed in the test split; must be within
            ``(0.0, 1.0)``.
        seed: Seed for the internal ``random.Random`` instance.
        stratify_by: Optional Case attribute name (``"expected"``) or
            ``metadata.<key>`` dotted path used to preserve class balance.

    Returns:
        ``(train_cases, test_cases)`` tuple.

    Raises:
        ValueError: If ``test_size`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0.0, 1.0), got {test_size}")

    rng = random.Random(seed)
    n = len(cases)
    if n == 0:
        return [], []

    test_idx: set[int]

    if stratify_by is None:
        indices = list(range(n))
        rng.shuffle(indices)
        n_test = max(1, int(round(n * test_size))) if n > 1 else 0
        n_test = min(n_test, n - 1) if n > 1 else 0
        test_idx = set(indices[:n_test])
        train = [cases[i] for i in range(n) if i not in test_idx]
        test = [cases[i] for i in range(n) if i in test_idx]
        return train, test

    buckets: dict[Any, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        buckets[_resolve_stratify_key(case, stratify_by)].append(i)

    test_idx = set()
    for key in sorted(buckets.keys(), key=lambda k: (k is None, str(k))):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        bucket_n = len(bucket)
        bucket_test = int(round(bucket_n * test_size))
        if bucket_n > 1:
            bucket_test = max(1, min(bucket_test, bucket_n - 1))
        else:
            bucket_test = 0
        test_idx.update(bucket[:bucket_test])

    train = [cases[i] for i in range(n) if i not in test_idx]
    test = [cases[i] for i in range(n) if i in test_idx]
    return train, test


def k_fold_split(
    cases: list[Case],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[list[Case], list[Case]]]:
    """Produce ``k`` reproducible ``(train, test)`` folds.

    Args:
        cases: The cases to split.
        k: Number of folds; must be ``>= 2`` and ``<= len(cases)``.
        seed: Seed for the internal ``random.Random`` instance.

    Returns:
        A list of ``k`` ``(train_cases, test_cases)`` tuples whose test splits
        are disjoint and together cover every input case exactly once.

    Raises:
        ValueError: If ``k`` is less than 2 or greater than ``len(cases)``.
    """
    n = len(cases)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed number of cases ({n})")

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds: list[list[int]] = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    result: list[tuple[list[Case], list[Case]]] = []
    for fold_idx in range(k):
        test_idx = set(folds[fold_idx])
        train = [cases[i] for i in range(n) if i not in test_idx]
        test = [cases[i] for i in range(n) if i in test_idx]
        result.append((train, test))
    return result


__all__: tuple[str, ...] = ("train_test_split", "k_fold_split")

# Mypy-friendly re-export hints that avoid shadowing stdlib ``random``.
_Splitter = Callable[..., tuple[list[Case], list[Case]]]
