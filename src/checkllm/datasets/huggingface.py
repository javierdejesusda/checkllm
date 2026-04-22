"""Hugging Face datasets integration for checkllm.

Provides helpers to load any Hugging Face dataset as a list of ``Case`` objects
and to push curated golden sets back to the Hub.  The ``datasets`` package is
an optional dependency; a helpful error is raised when it is missing.
"""

from __future__ import annotations

from typing import Any, Iterable

from checkllm.datasets.case import Case


_INPUT_FIELD_CANDIDATES: tuple[str, ...] = (
    "input",
    "question",
    "prompt",
    "instruction",
    "query",
    "text",
)

_EXPECTED_FIELD_CANDIDATES: tuple[str, ...] = (
    "expected",
    "answer",
    "label",
    "completion",
    "target",
    "output",
    "response",
)

_CONTEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "context",
    "passage",
    "document",
    "documents",
    "background",
)

_QUERY_FIELD_CANDIDATES: tuple[str, ...] = (
    "query",
    "question",
)

_CRITERIA_FIELD_CANDIDATES: tuple[str, ...] = (
    "criteria",
    "rubric",
)

_CASE_FIELDS: frozenset[str] = frozenset(
    {"input", "expected", "query", "context", "criteria", "metadata"}
)


def _require_datasets() -> Any:
    """Import and return the Hugging Face ``datasets`` module.

    Raises:
        ImportError: If ``datasets`` is not installed, with install guidance.
    """
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face integration. "
            "Install it with: pip install 'checkllm[hf]' or pip install datasets"
        ) from exc
    return datasets


def _auto_detect_field(row_keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in ``row_keys`` (case-insensitive)."""
    lowered = {k.lower(): k for k in row_keys}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _stringify(value: Any) -> str | None:
    """Convert arbitrary row values to a string suitable for Case fields."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(v) or "" for v in value)
    return str(value)


def _build_auto_field_map(row_keys: Iterable[str]) -> dict[str, str]:
    """Infer a sensible ``field_map`` from a row's keys."""
    keys = list(row_keys)
    mapping: dict[str, str] = {}

    input_src = _auto_detect_field(keys, _INPUT_FIELD_CANDIDATES)
    if input_src:
        mapping[input_src] = "input"

    expected_src = _auto_detect_field(keys, _EXPECTED_FIELD_CANDIDATES)
    if expected_src and expected_src != input_src:
        mapping[expected_src] = "expected"

    context_src = _auto_detect_field(keys, _CONTEXT_FIELD_CANDIDATES)
    if context_src and context_src not in mapping:
        mapping[context_src] = "context"

    query_src = _auto_detect_field(keys, _QUERY_FIELD_CANDIDATES)
    if query_src and query_src not in mapping:
        mapping[query_src] = "query"

    criteria_src = _auto_detect_field(keys, _CRITERIA_FIELD_CANDIDATES)
    if criteria_src and criteria_src not in mapping:
        mapping[criteria_src] = "criteria"

    return mapping


def _row_to_case(row: dict[str, Any], field_map: dict[str, str]) -> Case:
    """Convert a single HF dataset row into a ``Case`` using ``field_map``."""
    case_kwargs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    mapped_sources = set(field_map.keys())

    for src, target in field_map.items():
        if target not in _CASE_FIELDS:
            continue
        if src not in row:
            continue
        if target == "metadata":
            value = row[src]
            if isinstance(value, dict):
                metadata.update(value)
            else:
                metadata[src] = value
        else:
            case_kwargs[target] = _stringify(row[src])

    for key, value in row.items():
        if key in mapped_sources:
            continue
        metadata[key] = value

    if "input" not in case_kwargs:
        raise ValueError(
            "Could not determine the 'input' field for a row. "
            "Provide a field_map mapping one of the row's columns to 'input'. "
            f"Available columns: {sorted(row.keys())}"
        )

    if metadata:
        case_kwargs["metadata"] = metadata
    return Case(**case_kwargs)


def load_hf_dataset(
    name: str,
    split: str = "test",
    config: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
    field_map: dict[str, str] | None = None,
) -> list[Case]:
    """Load a Hugging Face dataset and return it as ``list[Case]``.

    Args:
        name: The Hugging Face dataset identifier (e.g. ``"squad"``).
        split: Dataset split to load (default ``"test"``).
        config: Optional config/subset name passed as ``name`` to
            ``datasets.load_dataset``'s second positional parameter.
        streaming: Load the dataset in streaming mode. Useful for very large
            datasets, combined with a ``limit`` to avoid exhausting memory.
        limit: Maximum number of rows to return.  If ``None`` the whole split
            is materialised.
        field_map: Optional mapping from HF column names to ``Case`` fields
            (``input``, ``expected``, ``query``, ``context``, ``criteria``,
            ``metadata``).  When omitted, common field names are auto-detected.

    Returns:
        A list of ``Case`` objects (never a generator).

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        ValueError: If no ``input`` column can be determined.
    """
    datasets = _require_datasets()
    ds = datasets.load_dataset(name, config, split=split, streaming=streaming)

    cases: list[Case] = []
    resolved_map: dict[str, str] | None = field_map

    iterable: Iterable[Any]
    if streaming:
        iterable = iter(ds)
    else:
        iterable = ds

    for index, raw_row in enumerate(iterable):
        if limit is not None and index >= limit:
            break
        if isinstance(raw_row, dict):
            row: dict[str, Any] = raw_row
        else:
            row = dict(raw_row)
        if resolved_map is None:
            resolved_map = _build_auto_field_map(row.keys())
            if not resolved_map:
                raise ValueError(
                    "Could not auto-detect any Case fields from columns "
                    f"{sorted(row.keys())}. Pass field_map explicitly."
                )
        cases.append(_row_to_case(row, resolved_map))

    return cases


def push_to_hub(
    cases: list[Case],
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    split: str = "test",
) -> str:
    """Push a list of ``Case`` objects to the Hugging Face Hub as a dataset.

    Args:
        cases: The cases to upload.
        repo_id: Target repository (e.g. ``"my-org/my-golden-set"``).
        token: Hugging Face auth token; falls back to the default cached token
            if ``None``.
        private: Whether to create the dataset repo as private.
        split: Split name to register the data under.

    Returns:
        The ``repo_id`` that was uploaded to.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    datasets = _require_datasets()
    records = [case.model_dump() for case in cases]
    ds = datasets.Dataset.from_list(records, split=split)
    ds.push_to_hub(repo_id, token=token, private=private)
    return repo_id


__all__ = ["load_hf_dataset", "push_to_hub"]
