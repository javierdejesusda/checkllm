"""Tests for judge auto-detection."""
from unittest.mock import patch
from checkllm.discovery import detect_judge_backend


def test_detect_openai_from_env():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        backend, model = detect_judge_backend()
        assert backend == "openai"


def test_detect_anthropic_when_no_openai():
    env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
    with patch.dict("os.environ", env, clear=True):
        backend, model = detect_judge_backend()
        assert backend == "anthropic"


def test_detect_gemini_when_no_openai_or_anthropic():
    env = {"GEMINI_API_KEY": "gem-test"}
    with patch.dict("os.environ", env, clear=True):
        backend, model = detect_judge_backend()
        assert backend == "gemini"


def test_detect_google_api_key_as_gemini():
    env = {"GOOGLE_API_KEY": "goog-test"}
    with patch.dict("os.environ", env, clear=True):
        backend, model = detect_judge_backend()
        assert backend == "gemini"


def test_detect_ollama_when_running(monkeypatch):
    """Detect Ollama when it's running on localhost."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with patch("checkllm.discovery._ollama_is_running", return_value=True):
        backend, model = detect_judge_backend()
        assert backend == "ollama"


def test_detect_returns_none_when_nothing_available(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with patch("checkllm.discovery._ollama_is_running", return_value=False):
        result = detect_judge_backend()
        assert result is None


def test_detect_respects_priority_order():
    """OpenAI wins when multiple keys are set."""
    env = {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "GEMINI_API_KEY": "gem-test",
    }
    with patch.dict("os.environ", env, clear=False):
        backend, model = detect_judge_backend()
        assert backend == "openai"


def test_default_models_per_backend():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        _, model = detect_judge_backend()
        assert model == "gpt-4o"

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"}, clear=True):
        _, model = detect_judge_backend()
        assert model == "claude-sonnet-4-6"

    with patch.dict("os.environ", {"GEMINI_API_KEY": "gem"}, clear=True):
        _, model = detect_judge_backend()
        assert model == "gemini-2.0-flash"
