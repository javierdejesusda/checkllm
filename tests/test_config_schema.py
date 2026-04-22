"""Tests for JSON-Schema-based checkllm.yaml validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkllm.config_schema import (
    ValidationError,
    generate_schema_to_file,
    load_schema,
    validate_config,
)


def test_schema_parses_as_valid_json() -> None:
    schema_path = (
        Path(__file__).parent.parent
        / "src"
        / "checkllm"
        / "schemas"
        / "checkllm.schema.json"
    )
    raw = schema_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["$schema"].endswith("draft-07/schema#")
    assert data["title"] == "CheckllmConfig"


def test_load_schema_returns_dict_with_properties() -> None:
    schema = load_schema()
    assert isinstance(schema, dict)
    props = schema["properties"]
    for expected in {"judge_model", "default_threshold", "max_concurrency", "engine", "providers"}:
        assert expected in props


def test_load_schema_returns_fresh_copy() -> None:
    a = load_schema()
    b = load_schema()
    a["properties"]["__mutation__"] = True
    assert "__mutation__" not in b["properties"]


def test_validate_empty_dict_has_no_required_errors() -> None:
    # The schema intentionally has no required keys at the top level —
    # every property has a safe default — so an empty dict is valid.
    errors = validate_config({})
    assert errors == []


def test_validate_accepts_known_good_config() -> None:
    good = {
        "judge_model": "gpt-4o",
        "default_threshold": 0.8,
        "runs_per_test": 1,
        "snapshot_dir": ".checkllm/snapshots",
        "cache_enabled": True,
        "cache_ttl_seconds": 604800,
        "max_concurrency": 10,
        "engine": "auto",
        "log_level": "WARNING",
    }
    errors = validate_config(good)
    assert errors == []


def test_validate_rejects_bad_threshold() -> None:
    errors = validate_config({"default_threshold": 1.7})
    assert errors
    assert any(e.severity == "error" and "default_threshold" in e.path for e in errors)


def test_validate_rejects_bad_engine_enum() -> None:
    errors = validate_config({"engine": "totally-made-up"})
    assert any(e.severity == "error" for e in errors)


def test_validate_accepts_unknown_keys_as_additional_properties() -> None:
    # additionalProperties: true at top level means unknown keys do not
    # produce errors.
    errors = validate_config({"totally_unknown": 42})
    assert all(e.severity != "error" or "unknown" not in e.message.lower() for e in errors)


def test_validate_rejects_wrong_type_for_max_concurrency() -> None:
    errors = validate_config({"max_concurrency": "lots"})
    assert errors
    assert any("max_concurrency" in e.path for e in errors)


def test_validate_accepts_providers_as_strings_or_objects() -> None:
    errors = validate_config(
        {
            "providers": [
                "openai:gpt-4o",
                {"id": "anthropic", "model": "claude-sonnet-4-6"},
            ]
        }
    )
    assert errors == []


def test_generate_schema_to_file_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "out" / "checkllm.schema.json"
    written = generate_schema_to_file(out)
    assert written == out
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["title"] == "CheckllmConfig"


def test_validate_error_model_fields() -> None:
    errors = validate_config({"default_threshold": -0.5})
    assert errors
    err = errors[0]
    assert isinstance(err, ValidationError)
    assert err.severity in {"error", "warning"}
    assert err.path.startswith("/")


def test_pyproject_checkllm_table_is_valid() -> None:
    """The live [tool.checkllm] table in pyproject.toml must validate."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    section = data.get("tool", {}).get("checkllm", {})
    # Drop the profiles sub-table since it uses its own shape.
    section_no_profiles = {k: v for k, v in section.items() if k != "profiles"}
    errors = validate_config(section_no_profiles)
    error_only = [e for e in errors if e.severity == "error"]
    assert error_only == [], f"Unexpected schema errors: {error_only}"
