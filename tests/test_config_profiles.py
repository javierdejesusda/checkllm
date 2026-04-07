"""Tests for configuration profile support."""
from __future__ import annotations

from pathlib import Path


from checkllm.config import CheckllmConfig, load_config


# ---------------------------------------------------------------------------
# New fields
# ---------------------------------------------------------------------------


class TestNewFields:
    def test_engine_default(self):
        config = CheckllmConfig()
        assert config.engine == "auto"

    def test_engine_custom(self):
        config = CheckllmConfig(engine="thread")
        assert config.engine == "thread"

    def test_active_profile_default(self):
        config = CheckllmConfig()
        assert config.active_profile is None

    def test_active_profile_set(self):
        config = CheckllmConfig(active_profile="dev")
        assert config.active_profile == "dev"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOML_WITH_PROFILES = """\
[tool.checkllm]
judge_model = "gpt-4o"
default_threshold = 0.8
log_level = "INFO"

[tool.checkllm.profiles.dev]
judge_model = "gpt-4o-mini"
budget = 1.0
log_level = "DEBUG"

[tool.checkllm.profiles.ci]
cache_enabled = false
budget = 10.0
log_level = "WARNING"

[tool.checkllm.profiles.prod]
judge_model = "gpt-4o"
default_threshold = 0.9
max_concurrency = 20
"""


def _write_toml(tmp_path: Path, content: str = _TOML_WITH_PROFILES) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


class TestProfileLoading:
    def test_no_profile_returns_base(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8
        assert config.log_level == "INFO"
        assert config.active_profile is None

    def test_dev_profile_overrides(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path, profile="dev")
        assert config.judge_model == "gpt-4o-mini"
        assert config.budget == 1.0
        assert config.log_level == "DEBUG"
        # Inherits base value
        assert config.default_threshold == 0.8
        assert config.active_profile == "dev"

    def test_ci_profile_overrides(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path, profile="ci")
        assert config.cache_enabled is False
        assert config.budget == 10.0
        assert config.log_level == "WARNING"
        # Inherits base values
        assert config.judge_model == "gpt-4o"
        assert config.active_profile == "ci"

    def test_prod_profile_overrides(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path, profile="prod")
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.9
        assert config.max_concurrency == 20
        assert config.active_profile == "prod"

    def test_unknown_profile_ignored(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path, profile="nonexistent")
        # Falls back to base config, but active_profile is still set
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8
        assert config.active_profile == "nonexistent"

    def test_no_profiles_section(self, tmp_path: Path):
        content = """\
[tool.checkllm]
judge_model = "gpt-4o-mini"
"""
        _write_toml(tmp_path, content)
        config = load_config(tmp_path, profile="dev")
        # No profiles defined - base config is used
        assert config.judge_model == "gpt-4o-mini"
        assert config.active_profile == "dev"


# ---------------------------------------------------------------------------
# Profile via env var
# ---------------------------------------------------------------------------


class TestProfileEnvVar:
    def test_env_var_selects_profile(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_PROFILE", "dev")
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o-mini"
        assert config.budget == 1.0
        assert config.active_profile == "dev"

    def test_parameter_takes_precedence_over_env(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_PROFILE", "dev")
        config = load_config(tmp_path, profile="ci")
        # profile= parameter wins
        assert config.cache_enabled is False
        assert config.budget == 10.0
        assert config.active_profile == "ci"

    def test_empty_env_var_ignored(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_PROFILE", "")
        config = load_config(tmp_path)
        # Empty string is falsy, no profile applied
        assert config.active_profile is None
        assert config.judge_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Merge precedence: defaults -> base -> profile -> env overrides
# ---------------------------------------------------------------------------


class TestMergePrecedence:
    def test_env_overrides_profile(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_JUDGE_MODEL", "gpt-3.5-turbo")
        config = load_config(tmp_path, profile="dev")
        # Profile says gpt-4o-mini, but env overrides it
        assert config.judge_model == "gpt-3.5-turbo"
        assert config.active_profile == "dev"
        # Profile budget is still applied (no env override for it)
        assert config.budget == 1.0

    def test_profile_overrides_base(self, tmp_path: Path):
        _write_toml(tmp_path)
        config = load_config(tmp_path, profile="prod")
        # Base says 0.8, prod says 0.9
        assert config.default_threshold == 0.9

    def test_full_precedence_chain(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_LOG_LEVEL", "ERROR")
        config = load_config(tmp_path, profile="dev")
        # Base says INFO, dev says DEBUG, env says ERROR -> env wins
        assert config.log_level == "ERROR"
        # Dev profile budget is kept (no env override)
        assert config.budget == 1.0
        assert config.active_profile == "dev"

    def test_engine_env_override(self, tmp_path: Path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.setenv("CHECKLLM_ENGINE", "process")
        config = load_config(tmp_path)
        assert config.engine == "process"


# ---------------------------------------------------------------------------
# Base config without profiles still works
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_existing_toml_without_profiles(self, tmp_path: Path):
        content = """\
[tool.checkllm]
judge_model = "gpt-4o"
default_threshold = 0.8
runs_per_test = 1
snapshot_dir = ".checkllm/snapshots"
cache_enabled = true
cache_ttl_seconds = 604800
max_concurrency = 10
log_level = "WARNING"
"""
        _write_toml(tmp_path, content)
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8
        assert config.active_profile is None
        assert config.engine == "auto"

    def test_no_pyproject_returns_defaults(self, tmp_path: Path):
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o"
        assert config.active_profile is None
        assert config.engine == "auto"
