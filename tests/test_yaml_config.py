"""Tests for checkllm.yaml_config — YAML declarative eval configuration."""

from __future__ import annotations

import pytest
import yaml

from checkllm.yaml_config import (
    AssertionConfig,
    EvalConfig,
    PromptConfig,
    ProviderConfig,
    TestConfig,
    load_eval_config,
    render_prompt,
)


# ---------------------------------------------------------------------------
# load_eval_config
# ---------------------------------------------------------------------------


class TestLoadEvalConfig:
    def test_loads_yaml(self, tmp_path):
        config_data = {
            "description": "Test evaluation suite",
            "providers": [{"id": "openai", "model": "gpt-4o", "options": {"temperature": 0.0}}],
            "prompts": [
                {
                    "name": "default",
                    "template": "Answer the question: {{input}}",
                    "variables": {"tone": "formal"},
                }
            ],
            "tests": [
                {
                    "description": "Basic test",
                    "input": "What is 2+2?",
                    "expected": "4",
                    "assertions": [
                        {"type": "contains", "value": "4"},
                        {"type": "max_tokens", "value": 100},
                    ],
                }
            ],
            "settings": {"max_concurrency": 3},
        }

        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_eval_config(yaml_file)

        assert isinstance(config, EvalConfig)
        assert config.description == "Test evaluation suite"

        assert len(config.providers) == 1
        assert config.providers[0].id == "openai"
        assert config.providers[0].model == "gpt-4o"
        assert config.providers[0].options["temperature"] == 0.0

        assert len(config.prompts) == 1
        assert config.prompts[0].name == "default"
        assert "{{input}}" in config.prompts[0].template
        assert config.prompts[0].variables["tone"] == "formal"

        assert len(config.tests) == 1
        assert config.tests[0].description == "Basic test"
        assert config.tests[0].input == "What is 2+2?"
        assert config.tests[0].expected == "4"
        assert len(config.tests[0].assertions) == 2
        assert config.tests[0].assertions[0].type == "contains"
        assert config.tests[0].assertions[0].value == "4"

        assert config.settings["max_concurrency"] == 3

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_eval_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_eval_config(bad_file)

    def test_non_mapping_yaml(self, tmp_path):
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_eval_config(list_file)

    def test_minimal_yaml(self, tmp_path):
        minimal_file = tmp_path / "minimal.yaml"
        minimal_file.write_text("description: minimal\n", encoding="utf-8")
        config = load_eval_config(minimal_file)
        assert config.description == "minimal"
        assert config.providers == []
        assert config.prompts == []
        assert config.tests == []


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_renders_variables(self):
        template = "Hello {{name}}, you asked: {{question}}"
        result = render_prompt(template, {"name": "Alice", "question": "What is AI?"})
        assert result == "Hello Alice, you asked: What is AI?"

    def test_no_variables(self):
        template = "This has no variables."
        result = render_prompt(template, {})
        assert result == "This has no variables."

    def test_renders_with_spaces_in_braces(self):
        template = "Hello {{ name }}"
        result = render_prompt(template, {"name": "Bob"})
        assert result == "Hello Bob"

    def test_missing_variable_left_as_empty(self):
        template = "Hello {{name}}"
        result = render_prompt(template, {})
        # Jinja2 renders missing variables as empty string by default
        assert result == "Hello "

    def test_multiple_occurrences(self):
        template = "{{x}} + {{x}} = {{result}}"
        result = render_prompt(template, {"x": "2", "result": "4"})
        assert result == "2 + 2 = 4"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_creates_config(self):
        cfg = ProviderConfig(id="openai", model="gpt-4o", api_key="sk-test")
        assert cfg.id == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "sk-test"
        assert cfg.options == {}

    def test_defaults(self):
        cfg = ProviderConfig(id="anthropic")
        assert cfg.model == ""
        assert cfg.api_key is None
        assert cfg.options == {}


class TestPromptConfig:
    def test_creates_config(self):
        cfg = PromptConfig(
            name="qa",
            template="Q: {{input}}\nA:",
            variables={"style": "concise"},
        )
        assert cfg.name == "qa"
        assert "{{input}}" in cfg.template
        assert cfg.variables["style"] == "concise"

    def test_defaults(self):
        cfg = PromptConfig(name="test", template="hello")
        assert cfg.variables == {}


class TestAssertionConfig:
    def test_creates_config(self):
        cfg = AssertionConfig(
            type="contains",
            value="expected text",
            threshold=0.8,
            options={"ignore_case": True},
        )
        assert cfg.type == "contains"
        assert cfg.value == "expected text"
        assert cfg.threshold == 0.8
        assert cfg.options["ignore_case"] is True

    def test_defaults(self):
        cfg = AssertionConfig(type="is_json")
        assert cfg.value is None
        assert cfg.threshold is None
        assert cfg.options == {}


class TestTestConfig:
    def test_creates_config(self):
        cfg = TestConfig(
            description="Math test",
            input="What is 2+2?",
            expected="4",
            context="arithmetic",
            query="2+2",
            assertions=[AssertionConfig(type="contains", value="4")],
            metadata={"difficulty": "easy"},
        )
        assert cfg.description == "Math test"
        assert cfg.input == "What is 2+2?"
        assert cfg.expected == "4"
        assert cfg.context == "arithmetic"
        assert cfg.query == "2+2"
        assert len(cfg.assertions) == 1
        assert cfg.metadata["difficulty"] == "easy"

    def test_defaults(self):
        cfg = TestConfig()
        assert cfg.description == ""
        assert cfg.input == ""
        assert cfg.expected is None
        assert cfg.context is None
        assert cfg.query is None
        assert cfg.assertions == []
        assert cfg.metadata == {}


class TestEvalConfig:
    def test_creates_config(self):
        cfg = EvalConfig(
            description="Full suite",
            providers=[ProviderConfig(id="openai")],
            prompts=[PromptConfig(name="default", template="{{input}}")],
            tests=[TestConfig(input="hello")],
            settings={"max_concurrency": 10},
        )
        assert cfg.description == "Full suite"
        assert len(cfg.providers) == 1
        assert len(cfg.prompts) == 1
        assert len(cfg.tests) == 1
        assert cfg.settings["max_concurrency"] == 10

    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.description == ""
        assert cfg.providers == []
        assert cfg.prompts == []
        assert cfg.tests == []
        assert cfg.settings == {}
