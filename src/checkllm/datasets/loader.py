from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, Sequence, Union

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


def load_dataset(
    source: Union[Path, str, Callable[..., Generator[Case, None, None]]]
) -> list[Case]:
    """Load a dataset from a YAML file path or a generator function."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            return load_yaml_dataset(path)
        raise ValueError(f"Unsupported dataset file format: {path.suffix}")
    if callable(source):
        return list(source())
    raise TypeError(f"Unsupported dataset source type: {type(source)}")
