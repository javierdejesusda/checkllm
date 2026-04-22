"""Auto-detect the best available judge backend from environment."""

from __future__ import annotations

import os
import socket


_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.0-flash",
    "cohere": "command-r-plus",
    "mistral": "mistral-large-latest",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "perplexity": "llama-3.1-sonar-large-128k-online",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "xai": "grok-2-latest",
    "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
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

    Priority order (first match wins):
        OpenAI > Anthropic > Gemini > Cohere > Mistral > DeepSeek > Groq >
        Together > Fireworks > Perplexity > OpenRouter > X.AI > Bedrock >
        Ollama (local).

    Returns ``(backend_name, default_model)`` or ``None`` if nothing found.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", _DEFAULT_MODELS["openai"])

    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", _DEFAULT_MODELS["anthropic"])

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return ("gemini", _DEFAULT_MODELS["gemini"])

    if os.environ.get("COHERE_API_KEY"):
        return ("cohere", _DEFAULT_MODELS["cohere"])

    if os.environ.get("MISTRAL_API_KEY"):
        return ("mistral", _DEFAULT_MODELS["mistral"])

    if os.environ.get("DEEPSEEK_API_KEY"):
        return ("deepseek", _DEFAULT_MODELS["deepseek"])

    if os.environ.get("GROQ_API_KEY"):
        return ("groq", _DEFAULT_MODELS["groq"])

    if os.environ.get("TOGETHER_API_KEY"):
        return ("together", _DEFAULT_MODELS["together"])

    if os.environ.get("FIREWORKS_API_KEY"):
        return ("fireworks", _DEFAULT_MODELS["fireworks"])

    if os.environ.get("PERPLEXITY_API_KEY"):
        return ("perplexity", _DEFAULT_MODELS["perplexity"])

    if os.environ.get("OPENROUTER_API_KEY"):
        return ("openrouter", _DEFAULT_MODELS["openrouter"])

    if os.environ.get("XAI_API_KEY"):
        return ("xai", _DEFAULT_MODELS["xai"])

    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        return ("bedrock", _DEFAULT_MODELS["bedrock"])

    if _ollama_is_running():
        return ("ollama", _DEFAULT_MODELS["ollama"])

    return None


def format_no_judge_error() -> str:
    """Return a helpful error message when no judge backend is available."""
    return (
        "No LLM judge backend found.\n\n"
        "checkllm looked for (in order):\n"
        "   1. OPENAI_API_KEY environment variable\n"
        "   2. ANTHROPIC_API_KEY environment variable\n"
        "   3. GEMINI_API_KEY or GOOGLE_API_KEY environment variable\n"
        "   4. COHERE_API_KEY environment variable\n"
        "   5. MISTRAL_API_KEY environment variable\n"
        "   6. DEEPSEEK_API_KEY environment variable\n"
        "   7. GROQ_API_KEY environment variable\n"
        "   8. TOGETHER_API_KEY environment variable\n"
        "   9. FIREWORKS_API_KEY environment variable\n"
        "  10. PERPLEXITY_API_KEY environment variable\n"
        "  11. OPENROUTER_API_KEY environment variable\n"
        "  12. XAI_API_KEY environment variable\n"
        "  13. AWS_ACCESS_KEY_ID or AWS_PROFILE (Bedrock)\n"
        "  14. Ollama running on localhost:11434\n\n"
        "To fix this, either:\n"
        "  - Set an API key:  export OPENAI_API_KEY=sk-...\n"
        "  - Start Ollama:    ollama serve  (free, local)\n"
        "  - Run 'checkllm init' to configure your project\n\n"
        "Tip: Deterministic checks like check.contains() work with no API key."
    )
