from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Generator, Union

import yaml

from checkllm.datasets.case import Case


def load_yaml_dataset(path: Path) -> list[Case]:
    """Load a dataset from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Dataset file must contain a YAML list, got {type(raw).__name__}")
    return [Case(**item) for item in raw]


def load_json_dataset(path: Path) -> list[Case]:
    """Load a dataset from a JSON file (array of objects)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Dataset file must contain a JSON array, got {type(raw).__name__}")
    return [Case(**item) for item in raw]


def load_csv_dataset(path: Path) -> list[Case]:
    """Load a dataset from a CSV file.

    The CSV must have a header row. The ``input`` column is required;
    ``expected``, ``query``, ``context``, and ``criteria`` are optional.
    Any extra columns are stored in the ``metadata`` field.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    case_fields = {"input", "expected", "query", "context", "criteria"}
    cases: list[Case] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_kwargs: dict[str, Any] = {}
            metadata: dict[str, Any] = {}
            for key, value in row.items():
                if key in case_fields:
                    case_kwargs[key] = value if value else None
                else:
                    metadata[key] = value
            if metadata:
                case_kwargs["metadata"] = metadata
            cases.append(Case(**case_kwargs))

    return cases


def load_dataset(
    source: Union[Path, str, Callable[..., Generator[Case, None, None]]]
) -> list[Case]:
    """Load a dataset from a file path (YAML, JSON, CSV) or a generator function."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            return load_yaml_dataset(path)
        if path.suffix == ".json":
            return load_json_dataset(path)
        if path.suffix == ".csv":
            return load_csv_dataset(path)
        raise ValueError(
            f"Unsupported dataset file format: {path.suffix}. "
            "Supported: .yaml, .yml, .json, .csv"
        )
    if callable(source):
        return list(source())
    raise TypeError(f"Unsupported dataset source type: {type(source)}")
