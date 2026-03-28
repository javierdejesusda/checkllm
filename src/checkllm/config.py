from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class CheckllmConfig(BaseModel):
    """Configuration for checkllm, loaded from pyproject.toml."""

    judge_backend: str = "openai"  # "openai" or "anthropic"
    judge_model: str = "gpt-4o"
    default_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    runs_per_test: int = Field(default=1, ge=1)
    snapshot_dir: str = ".checkllm/snapshots"
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    p_value_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("runs_per_test")
    @classmethod
    def validate_runs(cls, v: int) -> int:
        if v < 1:
            raise ValueError("runs_per_test must be >= 1")
        return v


def load_config(project_dir: Path | None = None) -> CheckllmConfig:
    """Load config from pyproject.toml [tool.checkllm], with env var overrides."""
    file_values: dict = {}

    if project_dir is None:
        project_dir = Path.cwd()

    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        file_values = data.get("tool", {}).get("checkllm", {})

    env_overrides: dict = {}
    env_map = {
        "CHECKLLM_JUDGE_BACKEND": "judge_backend",
        "CHECKLLM_JUDGE_MODEL": "judge_model",
        "CHECKLLM_DEFAULT_THRESHOLD": "default_threshold",
        "CHECKLLM_RUNS_PER_TEST": "runs_per_test",
        "CHECKLLM_SNAPSHOT_DIR": "snapshot_dir",
        "CHECKLLM_CONFIDENCE_LEVEL": "confidence_level",
        "CHECKLLM_P_VALUE_THRESHOLD": "p_value_threshold",
    }
    for env_key, config_key in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            env_overrides[config_key] = value

    merged = {**file_values, **env_overrides}
    return CheckllmConfig(**merged)
