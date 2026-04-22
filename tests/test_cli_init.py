"""Tests for the interactive checkllm init command."""

from unittest.mock import patch
from typer.testing import CliRunner
from checkllm.cli import app

runner = CliRunner()


def test_init_creates_test_file(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "general"])
    assert result.exit_code == 0
    tests_dir = tmp_path / "tests"
    assert tests_dir.exists()
    test_files = list(tests_dir.glob("test_*.py"))
    assert len(test_files) >= 1


def test_init_rag_generates_rag_checks(tmp_path):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
        result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "rag"])
    assert result.exit_code == 0
    test_file = tmp_path / "tests" / "test_llm_example.py"
    content = test_file.read_text()
    assert "hallucination" in content
    assert "faithfulness" in content


def test_init_chatbot_generates_chatbot_checks(tmp_path):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
        result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "chatbot"])
    assert result.exit_code == 0
    test_file = tmp_path / "tests" / "test_llm_example.py"
    content = test_file.read_text()
    assert "toxicity" in content
    assert "relevance" in content


def test_init_agent_generates_agent_checks(tmp_path):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
        result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "agent"])
    assert result.exit_code == 0
    test_file = tmp_path / "tests" / "test_llm_example.py"
    content = test_file.read_text()
    assert "task_completion" in content


def test_init_detects_api_keys(tmp_path):
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        with patch("checkllm.discovery._ollama_is_running", return_value=False):
            result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "general"])
            assert result.exit_code == 0
            pyproject = tmp_path / "pyproject.toml"
            content = pyproject.read_text()
            assert "anthropic" in content


def test_init_no_api_key_generates_deterministic_only(tmp_path):
    with patch.dict("os.environ", {}, clear=True):
        with patch("checkllm.discovery._ollama_is_running", return_value=False):
            result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "general"])
            assert result.exit_code == 0
            test_file = tmp_path / "tests" / "test_llm_example.py"
            content = test_file.read_text()
            assert "check.contains" in content


def test_init_with_ci_flag(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "general", "--ci"])
    assert result.exit_code == 0
    ci_file = tmp_path / ".github" / "workflows" / "checkllm.yml"
    assert ci_file.exists()
    content = ci_file.read_text()
    assert "checkllm" in content


def test_init_does_not_overwrite_existing_test(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    existing = tests_dir / "test_llm_example.py"
    existing.write_text("# my existing test\n")
    result = runner.invoke(app, ["init", str(tmp_path), "--use-case", "general"])
    assert result.exit_code == 0
    assert existing.read_text() == "# my existing test\n"
