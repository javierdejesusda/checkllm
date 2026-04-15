from __future__ import annotations

import json
from typing import Iterable

from bench.schema import BenchmarkSample, GroundTruth


def load_halubench_from_rows(rows: Iterable[dict]) -> list[BenchmarkSample]:
    """Convert raw HaluBench rows into BenchmarkSample objects.

    Args:
        rows: Iterable of dicts with keys id, question, passage, answer, label.

    Returns:
        List of BenchmarkSample with label 1.0 for PASS and 0.0 for FAIL.
    """
    out: list[BenchmarkSample] = []
    for row in rows:
        label = 1.0 if row["label"] == "PASS" else 0.0
        out.append(
            BenchmarkSample(
                sample_id=str(row["id"]),
                dataset="halubench",
                query=row["question"],
                answer=row["answer"],
                context=row["passage"],
                ground_truth=GroundTruth(label=label, kind="binary"),
            )
        )
    return out


def _ragtruth_has_hallucination(raw_labels) -> bool:
    """Return True when a RAGTruth row's hallucination_labels indicates a hallucination.

    The HuggingFace ``wandb/RAGTruth-processed`` dataset serializes
    ``hallucination_labels`` as a JSON string (e.g. ``"[]"`` or
    ``'[{"type": "evident_conflict", ...}]'``), so truthy-checking the raw
    value treats every row as hallucinated. Test fixtures, on the other hand,
    use real Python lists. This helper accepts both shapes.

    Args:
        raw_labels: The ``hallucination_labels`` field from a row.

    Returns:
        True when the label list is non-empty, False otherwise.
    """
    if raw_labels is None:
        return False
    if isinstance(raw_labels, str):
        try:
            parsed = json.loads(raw_labels)
        except json.JSONDecodeError:
            return False
        return bool(parsed)
    return bool(raw_labels)


def load_ragtruth_from_rows(rows: Iterable[dict]) -> list[BenchmarkSample]:
    """Convert raw RAGTruth rows into BenchmarkSample objects.

    Args:
        rows: Iterable of dicts with keys id, query, context, output,
            hallucination_labels.

    Returns:
        List of BenchmarkSample with label 1.0 when hallucination_labels is
        empty (faithful) and 0.0 when any label is present (hallucinated).
    """
    out: list[BenchmarkSample] = []
    for row in rows:
        has_hall = _ragtruth_has_hallucination(row.get("hallucination_labels"))
        label = 0.0 if has_hall else 1.0
        out.append(
            BenchmarkSample(
                sample_id=str(row["id"]),
                dataset="ragtruth",
                query=row["query"],
                answer=row["output"],
                context=row["context"],
                ground_truth=GroundTruth(label=label, kind="binary"),
            )
        )
    return out


def load_truthfulqa_from_rows(rows: Iterable[dict]) -> list[BenchmarkSample]:
    """Convert raw TruthfulQA rows into BenchmarkSample objects.

    The best_answer is used as both the answer and the context/reference since
    TruthfulQA is a reference-free evaluation setup. The HuggingFace dataset
    does not ship with stable ids, so we fall back to a positional sample_id
    when the row has no ``id`` field.

    Args:
        rows: Iterable of dicts with keys question, best_answer,
            correct_answers, incorrect_answers, and optionally id.

    Returns:
        List of BenchmarkSample with label 1.0 (scalar reference).
    """
    out: list[BenchmarkSample] = []
    for idx, row in enumerate(rows):
        raw_id = row.get("id") if "id" in row else None
        sample_id = f"tq-{idx}" if raw_id is None else str(raw_id)
        out.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset="truthfulqa",
                query=row["question"],
                answer=row["best_answer"],
                context=row["best_answer"],
                ground_truth=GroundTruth(label=1.0, kind="scalar"),
            )
        )
    return out


def load_jailbreakbench_from_rows(rows: Iterable[dict]) -> list[BenchmarkSample]:
    """Convert raw JailbreakBench rows into BenchmarkSample objects.

    Args:
        rows: Iterable of dicts with keys id, goal, category, behavior.

    Returns:
        List of BenchmarkSample with label 0.0 for harmful behaviors (should
        be refused) and 1.0 for benign behaviors.
    """
    out: list[BenchmarkSample] = []
    for row in rows:
        label = 0.0 if row["behavior"] == "harmful" else 1.0
        out.append(
            BenchmarkSample(
                sample_id=str(row["id"]),
                dataset="jailbreakbench",
                query=row["goal"],
                answer="",
                context="",
                ground_truth=GroundTruth(label=label, kind="binary"),
            )
        )
    return out


def load_hf(
    dataset_name: str,
    split: str = "test",
    limit: int | None = None,
    config: str | None = None,
    shuffle_seed: int | None = None,
) -> list[dict]:
    """Fetch a dataset from HuggingFace Hub and return rows as plain dicts.

    Lazily imports ``datasets`` so the module can be imported without it
    installed (tests use fixtures instead).

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``"truthful_qa"``).
        split: Dataset split to load (default ``"test"``).
        limit: Optional cap on the number of rows returned.
        config: Optional dataset config name for datasets that expose
            multiple subsets (e.g. ``"generation"`` for TruthfulQA).
        shuffle_seed: When set, shuffles the dataset with this seed before
            applying ``limit``. Needed for datasets whose test split is
            sorted by label (e.g. PatronusAI/HaluBench).

    Returns:
        List of dicts, one per row, up to ``limit`` rows.
    """
    from datasets import load_dataset  # noqa: PLC0415

    if config is not None:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]
