"""Auto-detect the best available judge backend from environment."""
from __future__ import annotations

import os
import socket


_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1",
}


def _ollama_is_running(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Check if Ollama is listening on localhost."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def detect_judge_backend() -> tuple[str, str] | None:
    """Detect the best available judge backend.

    Priority: OpenAI > Anthropic > Gemini > Ollama (local).
    Returns ``(backend_name, default_model)`` or ``None`` if nothing found.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", _DEFAULT_MODELS["openai"])

    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", _DEFAULT_MODELS["anthropic"])

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return ("gemini", _DEFAULT_MODELS["gemini"])

    if _ollama_is_running():
        return ("ollama", _DEFAULT_MODELS["ollama"])

    return None


def format_no_judge_error() -> str:
    """Return a helpful error message when no judge backend is available."""
    return (
        "No LLM judge backend found.\n\n"
        "checkllm looked for (in order):\n"
        "  1. OPENAI_API_KEY environment variable\n"
        "  2. ANTHROPIC_API_KEY environment variable\n"
        "  3. GEMINI_API_KEY or GOOGLE_API_KEY environment variable\n"
        "  4. Ollama running on localhost:11434\n\n"
        "To fix this, either:\n"
        "  - Set an API key:  export OPENAI_API_KEY=sk-...\n"
        "  - Start Ollama:    ollama serve  (free, local)\n"
        "  - Run 'checkllm init' to configure your project\n\n"
        "Tip: Deterministic checks like check.contains() work with no API key."
    )
