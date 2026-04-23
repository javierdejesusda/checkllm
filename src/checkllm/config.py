from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class CheckllmConfig(BaseModel):
    """Configuration for checkllm, loaded from pyproject.toml."""

    judge_backend: str = (
        "auto"  # "auto", "openai", "anthropic", "gemini", "azure", "ollama", "litellm"
    )
    judge_model: str = "gpt-4o"
    default_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    runs_per_test: int = Field(default=1, ge=1)
    snapshot_dir: str = ".checkllm/snapshots"
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    p_value_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # Caching
    cache_enabled: bool = True
    cache_dir: str = ".checkllm"
    cache_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=0)

    # Parallelism
    max_concurrency: int = Field(default=10, ge=1)

    # Cost budget (None = unlimited)
    budget: float | None = Field(default=None, ge=0.0)

    # Logging
    log_level: str = "WARNING"

    # Engine type
    engine: str = "auto"

    # Per-provider rate limits. Keys are provider names ("openai",
    # "anthropic", "bedrock", ...). Each value is ``{"rpm": int, "tpm": int}``.
    # Missing providers fall back to the defaults in
    # :mod:`checkllm.rate_limit`.
    rate_limits: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Retry policy knobs for 429-aware backoff in :class:`AsyncEngine`.
    retry_max_attempts: int = Field(default=5, ge=1)
    retry_base_delay: float = Field(default=1.0, ge=0.0)
    retry_max_delay: float = Field(default=60.0, ge=0.0)

    # Active profile (None when no profile is selected)
    active_profile: str | None = None

    def build_rate_limiter(self) -> Any:
        """Construct a :class:`ProviderRateLimiter` using ``rate_limits``.

        Returns a fresh limiter seeded with defaults plus any overrides
        declared in this config. Callers typically pass the result to
        :class:`checkllm.engines.AsyncEngine`.
        """
        from checkllm.rate_limit import ProviderRateLimiter, RateLimit

        overrides: dict[str, RateLimit] = {}
        for provider, cfg in self.rate_limits.items():
            rpm = int(cfg.get("rpm", 0))
            tpm = int(cfg.get("tpm", 0))
            if rpm <= 0 or tpm <= 0:
                continue
            overrides[provider] = RateLimit(rpm=rpm, tpm=tpm)
        return ProviderRateLimiter(limits=overrides)

    def build_retry_config(self) -> Any:
        """Construct a :class:`RetryConfig` from config fields."""
        from checkllm.rate_limit import RetryConfig

        return RetryConfig(
            max_attempts=self.retry_max_attempts,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    @field_validator("runs_per_test")
    @classmethod
    def validate_runs(cls, v: int) -> int:
        if v < 1:
            raise ValueError("runs_per_test must be >= 1")
        return v


def load_config(
    project_dir: Path | None = None,
    profile: str | None = None,
) -> CheckllmConfig:
    """Load config from pyproject.toml ``[tool.checkllm]``, with env var overrides.

    Merge order: defaults -> base config -> profile overrides -> env var overrides.

    Parameters
    ----------
    project_dir:
        Directory containing ``pyproject.toml``.  Defaults to ``Path.cwd()``.
    profile:
        Profile name to activate.  Resolution order:
        *profile* parameter > ``CHECKLLM_PROFILE`` env var > ``None``.
    """
    file_values: dict[str, Any] = {}
    profiles: dict[str, dict[str, Any]] = {}

    if project_dir is None:
        project_dir = Path.cwd()

    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        checkllm_section = data.get("tool", {}).get("checkllm", {})

        # Separate profiles from base config values
        for key, value in checkllm_section.items():
            if key == "profiles":
                if isinstance(value, dict):
                    profiles = value
            else:
                file_values[key] = value

    # Resolve which profile to use: parameter > env var > None
    resolved_profile = profile or os.environ.get("CHECKLLM_PROFILE") or None

    # Merge: base config <- profile overrides
    merged: dict[str, Any] = {**file_values}
    if resolved_profile and resolved_profile in profiles:
        merged.update(profiles[resolved_profile])

    # Apply env var overrides last
    env_overrides: dict[str, Any] = {}
    env_map = {
        "CHECKLLM_JUDGE_BACKEND": "judge_backend",
        "CHECKLLM_JUDGE_MODEL": "judge_model",
        "CHECKLLM_DEFAULT_THRESHOLD": "default_threshold",
        "CHECKLLM_RUNS_PER_TEST": "runs_per_test",
        "CHECKLLM_SNAPSHOT_DIR": "snapshot_dir",
        "CHECKLLM_CONFIDENCE_LEVEL": "confidence_level",
        "CHECKLLM_P_VALUE_THRESHOLD": "p_value_threshold",
        "CHECKLLM_CACHE_ENABLED": "cache_enabled",
        "CHECKLLM_CACHE_DIR": "cache_dir",
        "CHECKLLM_CACHE_TTL": "cache_ttl_seconds",
        "CHECKLLM_MAX_CONCURRENCY": "max_concurrency",
        "CHECKLLM_BUDGET": "budget",
        "CHECKLLM_LOG_LEVEL": "log_level",
        "CHECKLLM_ENGINE": "engine",
    }
    for env_key, config_key in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            env_overrides[config_key] = value

    merged.update(env_overrides)

    # Record which profile is active
    merged["active_profile"] = resolved_profile

    return CheckllmConfig(**merged)
