"""Golden-set lineage tracking backed by a JSON file.

Provides a ``DatasetVersion`` dataclass, a ``LineageStore`` that persists
versions to ``.checkllm/lineage.json``, and a ``track_lineage`` decorator that
automatically registers cases returned by loader functions.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from checkllm.datasets.case import Case


DEFAULT_LINEAGE_PATH = Path(".checkllm") / "lineage.json"


@dataclass
class DatasetVersion:
    """Metadata describing a snapshot of a registered dataset."""

    dataset_id: str
    version: str
    created_at: str
    content_hash: str
    parent_version: str | None = None
    num_cases: int = 0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the version record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVersion":
        """Rehydrate a ``DatasetVersion`` from a JSON-decoded dict."""
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            created_at=data["created_at"],
            content_hash=data["content_hash"],
            parent_version=data.get("parent_version"),
            num_cases=int(data.get("num_cases", 0)),
            source=data.get("source"),
            metadata=dict(data.get("metadata") or {}),
        )


def _case_to_sortable(case: Case) -> dict[str, Any]:
    """Normalise a Case for hashing; dumped via sorted-keys JSON."""
    return case.model_dump()


def compute_content_hash(cases: list[Case]) -> str:
    """Return the SHA256 hash of a list of cases.

    The hash is stable across runs: cases are serialised with sorted keys
    and without whitespace variation.  The order of cases in the input is
    preserved (order matters for dataset identity).
    """
    serialised = json.dumps(
        [_case_to_sortable(c) for c in cases],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _case_identity(case: Case) -> str:
    """Hash a single case for added/removed/modified detection."""
    payload = json.dumps(
        case.model_dump(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LineageDiff:
    """Summary of the difference between two dataset versions."""

    added: int = 0
    removed: int = 0
    modified: int = 0
    v1: str = ""
    v2: str = ""
    v1_hash: str = ""
    v2_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the diff."""
        return asdict(self)


class LineageStore:
    """JSON-backed store for ``DatasetVersion`` records.

    The store keeps all versions for every registered ``dataset_id`` in a
    single JSON file (default: ``.checkllm/lineage.json``).  Cases are
    persisted alongside the metadata so diffs can be computed later.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialise the store and ensure the parent directory exists.

        Args:
            path: Optional override path for the JSON file.
        """
        self.path: Path = Path(path) if path is not None else DEFAULT_LINEAGE_PATH
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"datasets": {}}
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"datasets": {}}
        if not isinstance(raw, dict) or "datasets" not in raw:
            return {"datasets": {}}
        return raw

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True, default=str)

    def _next_version(self, dataset_id: str) -> str:
        existing = self._data["datasets"].get(dataset_id, {}).get("versions", {})
        numbers: list[int] = []
        for v in existing:
            if v.startswith("v") and v[1:].isdigit():
                numbers.append(int(v[1:]))
        return f"v{(max(numbers) + 1) if numbers else 1}"

    def register(
        self,
        cases: list[Case],
        dataset_id: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Record a new version for ``dataset_id`` and return its metadata.

        Args:
            cases: The cases that make up this version.
            dataset_id: Human-readable identifier for the dataset
                (e.g. ``"qa-golden"``).
            source: Free-form provenance string (HF name, file path, ...).
            metadata: Additional metadata stored alongside the version.

        Returns:
            The newly created ``DatasetVersion``.  If the content hash matches
            the most-recent version for the same ``dataset_id``, that existing
            version is returned without duplicating storage.
        """
        content_hash = compute_content_hash(cases)
        ds_entry = self._data["datasets"].setdefault(dataset_id, {"versions": {}, "order": []})

        order: list[str] = ds_entry.setdefault("order", [])
        versions: dict[str, Any] = ds_entry.setdefault("versions", {})

        if order:
            latest_key = order[-1]
            latest = versions.get(latest_key)
            if latest and latest.get("content_hash") == content_hash:
                return DatasetVersion.from_dict(latest)

        parent_version = order[-1] if order else None
        version = self._next_version(dataset_id)
        dv = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            content_hash=content_hash,
            parent_version=parent_version,
            num_cases=len(cases),
            source=source,
            metadata=dict(metadata or {}),
        )

        record = dv.to_dict()
        record["cases"] = [c.model_dump() for c in cases]
        versions[version] = record
        order.append(version)
        self._save()
        return dv

    def get(self, dataset_id: str, version: str) -> DatasetVersion | None:
        """Return the stored metadata for ``(dataset_id, version)``."""
        ds = self._data["datasets"].get(dataset_id)
        if not ds:
            return None
        record = ds.get("versions", {}).get(version)
        if not record:
            return None
        return DatasetVersion.from_dict(record)

    def get_cases(self, dataset_id: str, version: str) -> list[Case] | None:
        """Return the cases stored for a specific version, if present."""
        ds = self._data["datasets"].get(dataset_id)
        if not ds:
            return None
        record = ds.get("versions", {}).get(version)
        if not record:
            return None
        return [Case(**c) for c in record.get("cases", [])]

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        """Return all versions for ``dataset_id`` in chronological order."""
        ds = self._data["datasets"].get(dataset_id)
        if not ds:
            return []
        order: list[str] = ds.get("order", [])
        versions: dict[str, Any] = ds.get("versions", {})
        result: list[DatasetVersion] = []
        for v in order:
            record = versions.get(v)
            if record:
                result.append(DatasetVersion.from_dict(record))
        return result

    def list_datasets(self) -> list[str]:
        """Return all registered dataset ids."""
        return sorted(self._data["datasets"].keys())

    def diff(
        self,
        v1: str,
        v2: str,
        dataset_id: str | None = None,
    ) -> LineageDiff:
        """Compute an added/removed/modified summary between two versions.

        Args:
            v1: Baseline version string.  May be prefixed with
                ``"<dataset_id>:"`` to disambiguate when ``dataset_id`` is
                not provided.
            v2: Target version string; same prefix rules apply.
            dataset_id: Explicit dataset id when ``v1``/``v2`` are bare
                version strings.

        Returns:
            A populated ``LineageDiff``.

        Raises:
            ValueError: If either version cannot be located in the store.
        """
        ds_v1, ver_v1 = self._split_version_ref(v1, dataset_id)
        ds_v2, ver_v2 = self._split_version_ref(v2, dataset_id)

        cases_v1 = self.get_cases(ds_v1, ver_v1)
        cases_v2 = self.get_cases(ds_v2, ver_v2)
        if cases_v1 is None:
            raise ValueError(f"Version not found: {ds_v1}:{ver_v1}")
        if cases_v2 is None:
            raise ValueError(f"Version not found: {ds_v2}:{ver_v2}")

        def _index_by_input(cases: list[Case]) -> dict[str, str]:
            return {c.input: _case_identity(c) for c in cases}

        idx1 = _index_by_input(cases_v1)
        idx2 = _index_by_input(cases_v2)

        inputs_v1 = set(idx1.keys())
        inputs_v2 = set(idx2.keys())

        added = len(inputs_v2 - inputs_v1)
        removed = len(inputs_v1 - inputs_v2)
        modified = sum(1 for key in inputs_v1 & inputs_v2 if idx1[key] != idx2[key])

        meta_v1 = self.get(ds_v1, ver_v1)
        meta_v2 = self.get(ds_v2, ver_v2)
        return LineageDiff(
            added=added,
            removed=removed,
            modified=modified,
            v1=f"{ds_v1}:{ver_v1}",
            v2=f"{ds_v2}:{ver_v2}",
            v1_hash=meta_v1.content_hash if meta_v1 else "",
            v2_hash=meta_v2.content_hash if meta_v2 else "",
        )

    def _split_version_ref(self, ref: str, fallback_dataset: str | None) -> tuple[str, str]:
        if ":" in ref:
            ds_id, version = ref.split(":", 1)
            return ds_id, version
        if fallback_dataset is None:
            raise ValueError(
                f"Ambiguous version reference {ref!r}; provide dataset_id "
                "or prefix with '<dataset_id>:<version>'."
            )
        return fallback_dataset, ref


def track_lineage(
    dataset_id: str,
    source: str | None = None,
    store: LineageStore | None = None,
) -> Callable[[Callable[..., list[Case]]], Callable[..., list[Case]]]:
    """Decorator that registers a loader's output with a ``LineageStore``.

    The decorator leaves the wrapped function's return value unchanged; it
    merely records a new version whenever the loader is called.  Lineage
    tracking can be disabled by setting ``CHECKLLM_LINEAGE_DISABLED=1``.

    Args:
        dataset_id: Dataset identifier under which to register versions.
        source: Free-form provenance string; defaults to the function name.
        store: Optional ``LineageStore`` instance; a default one is created
            on first call.

    Returns:
        A decorator that wraps a callable returning ``list[Case]``.
    """

    def decorator(func: Callable[..., list[Case]]) -> Callable[..., list[Case]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> list[Case]:
            cases = func(*args, **kwargs)
            if os.environ.get("CHECKLLM_LINEAGE_DISABLED") == "1":
                return cases
            try:
                active_store = store or LineageStore()
                active_store.register(
                    cases,
                    dataset_id=dataset_id,
                    source=source or func.__name__,
                )
            except Exception:
                # Never let lineage tracking break a data load.
                pass
            return cases

        return wrapper

    return decorator


__all__ = [
    "DatasetVersion",
    "LineageDiff",
    "LineageStore",
    "compute_content_hash",
    "track_lineage",
    "DEFAULT_LINEAGE_PATH",
]
