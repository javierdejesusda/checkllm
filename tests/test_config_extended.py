"""Tests for new configuration fields."""

import pytest

from checkllm.config import CheckllmConfig, load_config


class TestNewConfigFields:
    def test_defaults(self):
        config = CheckllmConfig()
        assert config.cache_enabled is True
        assert config.cache_dir == ".checkllm"
        assert config.cache_ttl_seconds == 7 * 24 * 3600
        assert config.max_concurrency == 10
        assert config.budget is None
        assert config.log_level == "WARNING"

    def test_budget_validation(self):
        config = CheckllmConfig(budget=5.0)
        assert config.budget == 5.0

    def test_budget_none(self):
        config = CheckllmConfig(budget=None)
        assert config.budget is None

    def test_max_concurrency_min(self):
        with pytest.raises(Exception):
            CheckllmConfig(max_concurrency=0)

    def test_cache_ttl_zero(self):
        config = CheckllmConfig(cache_ttl_seconds=0)
        assert config.cache_ttl_seconds == 0

    def test_env_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHECKLLM_CACHE_ENABLED", "false")
        monkeypatch.setenv("CHECKLLM_MAX_CONCURRENCY", "5")
        monkeypatch.setenv("CHECKLLM_BUDGET", "10.0")
        monkeypatch.setenv("CHECKLLM_LOG_LEVEL", "DEBUG")
        config = load_config(project_dir=tmp_path)
        assert config.cache_enabled is False
        assert config.max_concurrency == 5
        assert config.budget == 10.0
        assert config.log_level == "DEBUG"
