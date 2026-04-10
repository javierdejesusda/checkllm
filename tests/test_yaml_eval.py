"""Tests for checkllm.yaml_eval module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from checkllm.yaml_eval import (
    AssertionConfig,
    EvalSettings,
    JudgeConfig,
    EvalTestConfig,
    YAMLEvalConfig,
    YAMLEvalResult,
    YAMLEvaluator,
    _render_template,
    load_yaml_eval_config,
)


class TestYAMLEvalConfig:
    """Tests for the YAMLEvalConfig model."""

    def test_defaults(self):
        config = YAMLEvalConfig()
        assert config.description == ""
        assert config.judge.backend == "auto"
        assert config.judge.model == ""
        assert config.prompts == []
        assert config.providers == []
        assert config.tests == []
        assert config.settings.budget == 10.0
        assert config.settings.threshold == 0.8
        assert config.settings.parallel is True
        assert config.settings.cache is True

    def test_from_dict(self):
        config = YAMLEvalConfig(
            description="Test suite",
            judge=JudgeConfig(backend="openai", model="gpt-4o"),
            prompts=["Hello {{name}}"],
            providers=["openai:gpt-4o"],
            settings=EvalSettings(budget=5.0),
        )
        assert config.description == "Test suite"
        assert config.judge.backend == "openai"
        assert config.judge.model == "gpt-4o"
        assert len(config.prompts) == 1
        assert len(config.providers) == 1
        assert config.settings.budget == 5.0


class TestJudgeConfig:
    """Tests for JudgeConfig."""

    def test_defaults(self):
        cfg = JudgeConfig()
        assert cfg.backend == "auto"
        assert cfg.model == ""

    def test_custom(self):
        cfg = JudgeConfig(backend="anthropic", model="claude-sonnet-4-6")
        assert cfg.backend == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"


class TestAssertionConfig:
    """Tests for AssertionConfig."""

    def test_minimal(self):
        a = AssertionConfig(type="contains")
        assert a.type == "contains"
        assert a.value is None
        assert a.threshold is None

    def test_with_value(self):
        a = AssertionConfig(type="contains", value="hello")
        assert a.value == "hello"

    def test_with_threshold(self):
        a = AssertionConfig(type="relevance", threshold=0.9)
        assert a.threshold == 0.9


class TestEvalTestConfig:
    """Tests for EvalTestConfig."""

    def test_defaults(self):
        t = EvalTestConfig()
        assert t.vars == {}
        assert t.assert_ == []
        assert t.description == ""

    def test_with_vars_and_assertions(self):
        t = EvalTestConfig(
            vars={"query": "Hello"},
            assert_=[
                AssertionConfig(type="contains", value="hello"),
            ],
            description="Greeting test",
        )
        assert t.vars["query"] == "Hello"
        assert len(t.assert_) == 1
        assert t.description == "Greeting test"


class TestEvalSettings:
    """Tests for EvalSettings."""

    def test_defaults(self):
        s = EvalSettings()
        assert s.budget == 10.0
        assert s.threshold == 0.8
        assert s.parallel is True
        assert s.cache is True

    def test_custom(self):
        s = EvalSettings(budget=2.0, threshold=0.9, parallel=False, cache=False)
        assert s.budget == 2.0
        assert s.threshold == 0.9
        assert s.parallel is False
        assert s.cache is False


class TestRenderTemplate:
    """Tests for template rendering."""

    def test_simple_substitution(self):
        result = _render_template("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"

    def test_spaced_substitution(self):
        result = _render_template("Hello {{ name }}", {"name": "World"})
        assert result == "Hello World"

    def test_multiple_variables(self):
        result = _render_template(
            "{{greeting}} {{name}}!",
            {"greeting": "Hi", "name": "Alice"},
        )
        assert result == "Hi Alice!"

    def test_missing_variable_unchanged(self):
        result = _render_template("Hello {{name}}", {})
        assert result == "Hello {{name}}"

    def test_no_variables(self):
        result = _render_template("Hello World", {"name": "Foo"})
        assert result == "Hello World"


class TestLoadYamlEvalConfig:
    """Tests for loading YAML config files."""

    def test_loads_valid_yaml(self, tmp_path):
        config_data = {
            "description": "Test eval",
            "judge": {"backend": "openai", "model": "gpt-4o"},
            "prompts": ["Answer: {{query}}"],
            "providers": ["openai:gpt-4o"],
            "tests": [
                {
                    "vars": {"query": "What is 2+2?"},
                    "assert": [
                        {"type": "contains", "value": "4"},
                        {"type": "max_tokens", "value": 100},
                    ],
                }
            ],
            "settings": {"budget": 5.0, "threshold": 0.9},
        }

        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_yaml_eval_config(yaml_file)
        assert isinstance(config, YAMLEvalConfig)
        assert config.description == "Test eval"
        assert config.judge.backend == "openai"
        assert config.judge.model == "gpt-4o"
        assert len(config.prompts) == 1
        assert len(config.providers) == 1
        assert len(config.tests) == 1
        assert len(config.tests[0].assert_) == 2
        assert config.tests[0].vars["query"] == "What is 2+2?"
        assert config.settings.budget == 5.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml_eval_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("{{invalid yaml", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml_eval_config(yaml_file)

    def test_non_mapping_yaml(self, tmp_path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_yaml_eval_config(yaml_file)

    def test_minimal_config(self, tmp_path):
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text("{}\n", encoding="utf-8")
        config = load_yaml_eval_config(yaml_file)
        assert isinstance(config, YAMLEvalConfig)
        assert config.description == ""
        assert config.tests == []

    def test_multiple_tests_and_assertions(self, tmp_path):
        config_data = {
            "tests": [
                {
                    "vars": {"query": "q1"},
                    "assert": [
                        {"type": "contains", "value": "a"},
                        {"type": "no_pii"},
                    ],
                },
                {
                    "vars": {"query": "q2"},
                    "assert": [
                        {"type": "max_tokens", "value": 200},
                    ],
                },
            ],
        }

        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_yaml_eval_config(yaml_file)
        assert len(config.tests) == 2
        assert len(config.tests[0].assert_) == 2
        assert len(config.tests[1].assert_) == 1
        assert config.tests[0].assert_[1].type == "no_pii"


class TestYAMLEvalResult:
    """Tests for YAMLEvalResult."""

    def test_summary_all_pass(self):
        result = YAMLEvalResult(
            config_path="test.yaml",
            description="Test suite",
            total_tests=2,
            passed=2,
            failed=0,
            results=[],
            cost=0.01,
            duration_ms=500.0,
        )
        summary = result.summary()
        assert "2/2 passed" in summary
        assert "100%" in summary
        assert "$0.0100" in summary

    def test_summary_with_failures(self):
        result = YAMLEvalResult(
            config_path="test.yaml",
            description="",
            total_tests=3,
            passed=1,
            failed=2,
            results=[],
            cost=0.0,
            duration_ms=100.0,
        )
        summary = result.summary()
        assert "Failed: 2" in summary


class TestYAMLEvaluator:
    """Tests for the YAMLEvaluator class."""

    def test_load_config(self, tmp_path):
        config_data = {
            "description": "Evaluator test",
            "tests": [
                {
                    "vars": {"query": "test"},
                    "assert": [{"type": "contains", "value": "test"}],
                }
            ],
        }

        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        evaluator = YAMLEvaluator()
        config = evaluator.load_config(yaml_file)
        assert isinstance(config, YAMLEvalConfig)
        assert config.description == "Evaluator test"

    def test_run_deterministic_assertions(self, tmp_path):
        """Test running deterministic assertions that do not need an LLM."""
        config = YAMLEvalConfig(
            description="Deterministic test",
            prompts=["{{query}}"],
            providers=["default"],
            tests=[
                EvalTestConfig(
                    vars={"query": "What is 2+2?"},
                    assert_=[
                        AssertionConfig(type="contains", value="4"),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="The answer is 4.",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert isinstance(result, YAMLEvalResult)
        assert result.total_tests == 1
        assert result.passed == 1
        assert result.failed == 0
        assert len(result.results) == 1
        assert result.results[0].passed is True

    def test_run_contains_failure(self, tmp_path):
        """Test that a failing 'contains' assertion is correctly reported."""
        config = YAMLEvalConfig(
            description="Fail test",
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[
                        AssertionConfig(type="contains", value="nonexistent_string"),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="Hello world",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.total_tests == 1
        assert result.passed == 0
        assert result.failed == 1
        assert result.results[0].passed is False

    def test_run_no_pii_assertion(self, tmp_path):
        """Test the no_pii deterministic assertion."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[
                        AssertionConfig(type="no_pii"),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="This is a clean response with no personal data.",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.passed == 1
        assert result.failed == 0

    def test_run_max_tokens_assertion(self, tmp_path):
        """Test the max_tokens deterministic assertion."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[
                        AssertionConfig(type="max_tokens", value=10),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="Short response.",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.passed == 1

    def test_run_multiple_assertions(self, tmp_path):
        """Test running multiple assertions on one test case."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "price check"},
                    assert_=[
                        AssertionConfig(type="contains", value="price"),
                        AssertionConfig(type="no_pii"),
                        AssertionConfig(type="max_tokens", value=100),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="The price is $9.99.",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.total_tests == 3
        assert result.passed == 3
        assert result.failed == 0

    def test_run_multiple_test_cases(self, tmp_path):
        """Test running multiple test cases."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test1"},
                    assert_=[AssertionConfig(type="contains", value="hello")],
                ),
                EvalTestConfig(
                    vars={"query": "test2"},
                    assert_=[AssertionConfig(type="contains", value="world")],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="hello world",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.total_tests == 2
        assert result.passed == 2
        assert len(result.results) == 2

    def test_variable_substitution_in_prompt(self, tmp_path):
        """Test that variables are correctly substituted into prompts."""
        config = YAMLEvalConfig(
            prompts=["Answer this: {{query}}"],
            tests=[
                EvalTestConfig(
                    vars={"query": "What is AI?"},
                    assert_=[AssertionConfig(type="contains", value="AI")],
                ),
            ],
        )

        evaluator = YAMLEvaluator()
        captured_prompts = []

        async def mock_generate(prompt, provider_str, judge_config):
            captured_prompts.append(prompt)
            return "AI is artificial intelligence."

        with patch.object(evaluator, "_generate_output", side_effect=mock_generate):
            asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert len(captured_prompts) == 1
        assert captured_prompts[0] == "Answer this: What is AI?"

    def test_unknown_assertion_type(self, tmp_path):
        """Test that unknown assertion types result in failure."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[AssertionConfig(type="nonexistent_check")],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="output",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.failed == 1
        assert result.results[0].passed is False

    def test_llm_assertion_without_judge(self, tmp_path):
        """Test that LLM assertions fail gracefully without a judge."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[AssertionConfig(type="relevance", threshold=0.8)],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="some output",
        ), patch.object(
            evaluator,
            "_create_judge",
            return_value=None,
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.failed == 1
        assertion_result = result.results[0].assertions[0]
        assert "judge required" in assertion_result["reasoning"].lower()

    def test_result_cost_accumulation(self, tmp_path):
        """Test that costs are accumulated across assertions."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[
                        AssertionConfig(type="contains", value="a"),
                        AssertionConfig(type="contains", value="b"),
                    ],
                ),
            ],
        )

        evaluator = YAMLEvaluator()

        with patch.object(
            evaluator,
            "_generate_output",
            new_callable=AsyncMock,
            return_value="a and b",
        ):
            result = asyncio.new_event_loop().run_until_complete(
                evaluator.run_from_config(config)
            )

        assert result.cost == 0.0
        assert result.duration_ms > 0
