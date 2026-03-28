import os
from pathlib import Path

from checkllm.config import CheckllmConfig, load_config


class TestCheckllmConfig:
    def test_default_config(self):
        config = CheckllmConfig()
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8
        assert config.runs_per_test == 1
        assert config.snapshot_dir == ".checkllm/snapshots"
        assert config.confidence_level == 0.95
        assert config.p_value_threshold == 0.05

    def test_custom_config(self):
        config = CheckllmConfig(
            judge_model="gpt-4o-mini",
            default_threshold=0.9,
            runs_per_test=5,
        )
        assert config.judge_model == "gpt-4o-mini"
        assert config.default_threshold == 0.9
        assert config.runs_per_test == 5

    def test_runs_per_test_must_be_positive(self):
        import pytest

        with pytest.raises(ValueError):
            CheckllmConfig(runs_per_test=0)


class TestLoadConfig:
    def test_load_from_pyproject_toml(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.checkllm]\n'
            'judge_model = "gpt-4o-mini"\n'
            'default_threshold = 0.9\n'
            'runs_per_test = 5\n'
            'snapshot_dir = "my_snapshots"\n'
        )
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o-mini"
        assert config.default_threshold == 0.9
        assert config.runs_per_test == 5
        assert config.snapshot_dir == "my_snapshots"

    def test_load_returns_defaults_when_no_pyproject(self, tmp_path: Path):
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o"
        assert config.default_threshold == 0.8

    def test_load_returns_defaults_when_no_checkllm_section(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.other]\nkey = "value"\n')
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-4o"

    def test_load_config_with_env_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("CHECKLLM_JUDGE_MODEL", "gpt-3.5-turbo")
        config = load_config(tmp_path)
        assert config.judge_model == "gpt-3.5-turbo"
