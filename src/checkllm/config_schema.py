"""JSON Schema validation for checkllm configuration files.

This module wraps the bundled draft-07 JSON Schema describing the
``[tool.checkllm]`` table and the top-level keys accepted by
``checkllm.yaml``. It provides a small, dependency-optional validator
so that IDE tooling and CI jobs can catch typos and illegal values
without needing to boot the full config loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "checkllm.schema.json"


class ValidationError(BaseModel):
    """A single schema validation issue.

    Attributes:
        path: JSON pointer-style path into the document, e.g. ``"/judge_model"``.
        message: Human-readable explanation of what went wrong.
        severity: Either ``"error"`` or ``"warning"``.
    """

    path: str
    message: str
    severity: str = "error"


def load_schema() -> dict[str, Any]:
    """Return the bundled JSON schema as a Python dict.

    Returns:
        The parsed schema. A fresh dict is returned on each call so
        callers can mutate it without affecting future loads.

    Raises:
        FileNotFoundError: If the bundled schema file has been removed.
        ValueError: If the schema file is not valid JSON.
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Bundled schema missing at {_SCHEMA_PATH}")
    try:
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bundled schema is not valid JSON: {exc}") from exc


def _format_path(path_parts: list[Any]) -> str:
    """Render a jsonschema error path as a slash-delimited pointer."""
    if not path_parts:
        return "/"
    return "/" + "/".join(str(p) for p in path_parts)


def validate_config(data: dict[str, Any]) -> list[ValidationError]:
    """Validate *data* against the bundled checkllm JSON schema.

    Args:
        data: The parsed config dictionary (from ``pyproject.toml`` or
            ``checkllm.yaml``).

    Returns:
        A list of :class:`ValidationError` objects. An empty list
        indicates the document is valid. If the optional ``jsonschema``
        dependency is not installed the list contains a single error
        with installation instructions.
    """
    try:
        import jsonschema
        from jsonschema import Draft7Validator
    except ImportError:
        return [
            ValidationError(
                path="/",
                message=(
                    "jsonschema not installed — run `pip install jsonschema` "
                    "to enable config validation."
                ),
                severity="error",
            )
        ]

    schema = load_schema()
    validator = Draft7Validator(schema)
    errors: list[ValidationError] = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        severity = "error"
        # Unknown properties only surface if additionalProperties is false
        # anywhere in the schema; here we treat "additionalProperties"
        # violations as warnings.
        if err.validator == "additionalProperties":
            severity = "warning"
        errors.append(
            ValidationError(
                path=_format_path(list(err.absolute_path)),
                message=err.message,
                severity=severity,
            )
        )
    return errors


def generate_schema_to_file(path: Path) -> Path:
    """Write the bundled schema verbatim to *path*.

    Useful for CI jobs that want to keep a published copy of the schema
    in a repo root or docs site in sync with the package.

    Args:
        path: Destination file path. Parent directories are created.

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ValidationError",
    "generate_schema_to_file",
    "load_schema",
    "validate_config",
]
