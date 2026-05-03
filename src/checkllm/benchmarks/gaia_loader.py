"""Gated loader for the GAIA benchmark (Mialon et al. 2023).

GAIA (https://huggingface.co/datasets/gaia-benchmark/GAIA) is a gated
dataset. Users must accept the license on HuggingFace, provide an
``HF_TOKEN`` with access, and explicitly acknowledge the license in
CheckLLM by setting ``CHECKLLM_GAIA_LICENSE_ACK=yes``.

The loader pins a specific dataset revision SHA so that benchmark runs
are reproducible even if the upstream dataset is updated. The pinned SHA
lives in ``benchmarks/paper/dataset_manifest.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Indirection so tests can monkeypatch without importing the heavy `datasets` lib
# at module load time. Real callers get the real implementation lazily.
#
# Test seam. MUST only be overridden via ``pytest.MonkeyPatch.setattr`` (or an
# equivalent scoped-teardown mechanism). Direct module-level assignment has
# no reset and will pollute subsequent tests in the same process.
_hf_load_dataset = None


def _load_hf_real(
    name: str,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Any:
    """Thin wrapper around ``datasets.load_dataset``; imported lazily.

    Args:
        name: HuggingFace dataset repo id (e.g. ``"gaia-benchmark/GAIA"``).
        config: Dataset configuration name.
        split: Split to load (``"validation"`` or ``"test"``).
        revision: Pinned git revision SHA on the HF dataset repo.
        token: HuggingFace access token for gated datasets.

    Returns:
        The raw dataset object returned by ``datasets.load_dataset``.
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "revision": revision}
    if config is not None:
        kwargs["name"] = config
    if token is not None:
        kwargs["token"] = token
    return load_dataset(name, **kwargs)


_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "paper" / "dataset_manifest.json"
)

_VALID_SPLITS = frozenset({"validation", "test"})


class LicenseAcknowledgmentRequired(RuntimeError):
    """Raised when the GAIA license has not been acknowledged by the caller."""


class GaiaTask(BaseModel):
    """One GAIA question and its expected answer.

    Attributes:
        task_id: GAIA task identifier (stable across splits).
        question: Natural-language prompt shown to the agent.
        expected_answer: Exact-match gold answer used for scoring.
        level: GAIA difficulty level as a string ("1", "2", or "3").
        file_name: Relative path to an attached file, or ``None`` if
            the task is text-only.
    """

    task_id: str
    question: str
    expected_answer: str
    level: str
    file_name: str | None = Field(default=None)


def _require_license_ack() -> None:
    """Raise if ``CHECKLLM_GAIA_LICENSE_ACK`` is not ``"yes"``.

    Raises:
        LicenseAcknowledgmentRequired: If the env var is missing or not
            equal to ``"yes"`` (case-insensitive, whitespace-stripped).
    """
    value = os.environ.get("CHECKLLM_GAIA_LICENSE_ACK", "").strip().lower()
    if value != "yes":
        raise LicenseAcknowledgmentRequired(
            "GAIA is a gated dataset. Set CHECKLLM_GAIA_LICENSE_ACK=yes after "
            "accepting the license at https://huggingface.co/datasets/gaia-benchmark/GAIA "
            "and ensure HF_TOKEN is available in the environment."
        )


def _pinned_revision() -> str:
    """Return the pinned GAIA revision SHA from the paper dataset manifest.

    Returns:
        The 40-character git SHA pinned for the GAIA dataset.

    Raises:
        FileNotFoundError: If the manifest file is missing.
        KeyError: If the manifest has no ``gaia.revision`` entry.
    """
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Paper dataset manifest not found at {_MANIFEST_PATH}. "
            "This file pins dataset SHAs for reproducibility."
        )
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    revision = manifest.get("gaia", {}).get("revision")
    if not revision:
        raise KeyError(f"Missing manifest entry for gaia.revision at {_MANIFEST_PATH}")
    return str(revision)


def load_gaia(split: str, limit: int | None = None) -> list[GaiaTask]:
    """Load GAIA tasks for the given split, respecting the license gate.

    Args:
        split: Either ``"validation"`` or ``"test"``.
        limit: If not ``None``, return only the first ``limit`` tasks.

    Returns:
        A list of :class:`GaiaTask` instances.

    Raises:
        LicenseAcknowledgmentRequired: If ``CHECKLLM_GAIA_LICENSE_ACK``
            is not set to ``"yes"``.
        ValueError: If ``split`` is not one of ``"validation"``, ``"test"``.
        FileNotFoundError: If the paper dataset manifest is missing.
    """
    _require_license_ack()
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(_VALID_SPLITS)}")

    loader = _hf_load_dataset if _hf_load_dataset is not None else _load_hf_real
    revision = _pinned_revision()
    token = os.environ.get("HF_TOKEN")

    raw_rows = loader(
        "gaia-benchmark/GAIA",
        config="2023_all",
        split=split,
        revision=revision,
        token=token,
    )

    tasks: list[GaiaTask] = []
    for row in raw_rows:
        tasks.append(
            GaiaTask(
                task_id=str(row["task_id"]),
                question=str(row.get("Question", "")),
                expected_answer=str(row.get("Final answer", "")),
                level=str(row.get("Level", "")),
                file_name=row.get("file_name") or None,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
