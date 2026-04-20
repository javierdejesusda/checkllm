"""Comprehensive tests for checkllm.config — CheckllmConfig and load_config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from checkllm.config import CheckllmConfig, load_config


class TestCheckllmConfig:
    def test_default_values(self):
        config = CheckllmConfig()
        assert config.judge_backend == "auto"
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8
        assert config.runs_per_test == 1
        assert config.snapshot_dir == ".checkllm/snapshots"
        assert config.confidence_level == 0.95
        assert config.p_value_threshold == 0.05
        assert config.cache_enabled is True
        assert config.cache_dir == ".checkllm"
        assert config.max_concurrency == 10
        assert config.budget is None
        assert config.log_level == "WARNING"
        assert config.engine == "auto"
        assert config.active_profile is None

    def test_custom_values(self):
        config = CheckllmConfig(
            judge_backend="anthropic",
            judge_model="claude-3-5-sonnet-20241022",
            default_threshold=0.9,
            runs_per_test=3,
            max_concurrency=5,
            budget=10.0,
            log_level="DEBUG",
        )
        assert config.judge_backend == "anthropic"
        assert config.judge_model == "claude-3-5-sonnet-20241022"
        assert config.default_threshold == 0.9
        assert config.runs_per_test == 3
        assert config.max_concurrency == 5
        assert config.budget == 10.0
        assert config.log_level == "DEBUG"

    def test_runs_per_test_validation_passes(self):
        config = CheckllmConfig(runs_per_test=5)
        assert config.runs_per_test == 5

    def test_runs_per_test_validation_fails(self):
        with pytest.raises(Exception):
            CheckllmConfig(runs_per_test=0)

    def test_threshold_bounds(self):
        config = CheckllmConfig(default_threshold=0.0)
        assert config.default_threshold == 0.0

        config = CheckllmConfig(default_threshold=1.0)
        assert config.default_threshold == 1.0

    def test_threshold_out_of_bounds(self):
        with pytest.raises(Exception):
            CheckllmConfig(default_threshold=1.5)

    def test_cache_ttl_seconds(self):
        config = CheckllmConfig(cache_ttl_seconds=3600)
        assert config.cache_ttl_seconds == 3600

    def test_active_profile(self):
        config = CheckllmConfig(active_profile="ci")
        assert config.active_profile == "ci"


class TestLoadConfig:
    def test_defaults_when_no_toml(self, tmp_path: Path):
        config = load_config(project_dir=tmp_path)
        assert config.judge_backend == "auto"
        assert config.judge_model == "gpt-4o"

    def test_loads_from_pyproject_toml(self, tmp_path: Path):
        toml_content = """
[tool.checkllm]
judge_backend = "anthropic"
judge_model = "claude-3-5-haiku-20241022"
default_threshold = 0.85
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        config = load_config(project_dir=tmp_path)
        assert config.judge_backend == "anthropic"
        assert config.judge_model == "claude-3-5-haiku-20241022"
        assert config.default_threshold == pytest.approx(0.85)

    def test_env_var_overrides_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml_content = """
[tool.checkllm]
judge_model = "gpt-4o"
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        monkeypatch.setenv("CHECKLLM_JUDGE_MODEL", "gpt-4o-mini")
        config = load_config(project_dir=tmp_path)
        assert config.judge_model == "gpt-4o-mini"

    def test_env_var_overrides_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_JUDGE_BACKEND", "gemini")
        config = load_config(project_dir=tmp_path)
        assert config.judge_backend == "gemini"

    def test_env_var_log_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_LOG_LEVEL", "DEBUG")
        config = load_config(project_dir=tmp_path)
        assert config.log_level == "DEBUG"

    def test_env_var_engine(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_ENGINE", "openai")
        config = load_config(project_dir=tmp_path)
        assert config.engine == "openai"

    def test_profile_from_toml(self, tmp_path: Path):
        toml_content = """
[tool.checkllm]
judge_model = "gpt-4o"

[tool.checkllm.profiles.ci]
judge_model = "gpt-4o-mini"
max_concurrency = 3
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        config = load_config(project_dir=tmp_path, profile="ci")
        assert config.judge_model == "gpt-4o-mini"
        assert config.max_concurrency == 3
        assert config.active_profile == "ci"

    def test_profile_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml_content = """
[tool.checkllm]
judge_model = "gpt-4o"

[tool.checkllm.profiles.staging]
judge_model = "gpt-4o-mini"
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        monkeypatch.setenv("CHECKLLM_PROFILE", "staging")
        config = load_config(project_dir=tmp_path)
        assert config.judge_model == "gpt-4o-mini"
        assert config.active_profile == "staging"

    def test_profile_param_overrides_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml_content = """
[tool.checkllm]
judge_model = "gpt-4o"

[tool.checkllm.profiles.profile_a]
judge_model = "model-a"

[tool.checkllm.profiles.profile_b]
judge_model = "model-b"
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        monkeypatch.setenv("CHECKLLM_PROFILE", "profile_b")
        config = load_config(project_dir=tmp_path, profile="profile_a")
        assert config.judge_model == "model-a"
        assert config.active_profile == "profile_a"

    def test_unknown_profile_falls_back_to_base(self, tmp_path: Path):
        toml_content = """
[tool.checkllm]
judge_model = "gpt-4o"
"""
        (tmp_path / "pyproject.toml").write_text(toml_content, encoding="utf-8")
        config = load_config(project_dir=tmp_path, profile="nonexistent_profile")
        assert config.judge_model == "gpt-4o"
        assert config.active_profile == "nonexistent_profile"

    def test_no_active_profile_when_none_specified(self, tmp_path: Path):
        config = load_config(project_dir=tmp_path)
        assert config.active_profile is None

    def test_env_var_cache_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_CACHE_ENABLED", "false")
        config = load_config(project_dir=tmp_path)
        assert config.cache_enabled == "false" or config.cache_enabled is False  # string or converted

    def test_env_var_budget(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_BUDGET", "5.0")
        config = load_config(project_dir=tmp_path)
        assert config.budget is not None

    def test_env_var_max_concurrency(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHECKLLM_MAX_CONCURRENCY", "20")
        config = load_config(project_dir=tmp_path)
        # Should be applied
        assert config.max_concurrency == 20 or str(config.max_concurrency) == "20"
