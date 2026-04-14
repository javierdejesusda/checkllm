from __future__ import annotations

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
        has_hall = bool(row.get("hallucination_labels"))
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
    TruthfulQA is a reference-free evaluation setup.

    Args:
        rows: Iterable of dicts with keys id, question, best_answer,
            correct_answers, incorrect_answers.

    Returns:
        List of BenchmarkSample with label 1.0 (scalar reference).
    """
    out: list[BenchmarkSample] = []
    for row in rows:
        out.append(
            BenchmarkSample(
                sample_id=str(row["id"]),
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
) -> list[dict]:
    """Fetch a dataset from HuggingFace Hub and return rows as plain dicts.

    Lazily imports ``datasets`` so the module can be imported without it
    installed (tests use fixtures instead).

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``"truthful_qa"``).
        split: Dataset split to load (default ``"test"``).
        limit: Optional cap on the number of rows returned.

    Returns:
        List of dicts, one per row, up to ``limit`` rows.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]
