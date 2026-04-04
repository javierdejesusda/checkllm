"""User-friendly error messages for common failure scenarios."""
from __future__ import annotations


_EXTRA_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "gemini": "gemini",
    "litellm": "litellm",
    "embeddings": "embeddings",
    "sentence-transformers": "embeddings",
    "google-generativeai": "gemini",
}


def format_budget_error(
    budget: float,
    spent: float,
    completed: int,
    total: int,
) -> str:
    """Format a budget-exceeded error with actionable guidance."""
    return (
        f"Budget of ${budget:.2f} exhausted after {completed}/{total} checks "
        f"(spent ${spent:.4f}).\n\n"
        f"To fix this:\n"
        f"  - Increase budget:  checkllm run tests/ --budget {budget * 2:.0f}\n"
        f"  - Use a cheaper model:  judge_model = \"gpt-4o-mini\" in pyproject.toml\n"
        f"  - Replace judge checks with deterministic alternatives where possible"
    )


def format_missing_dependency_error(extra_name: str, class_name: str) -> str:
    """Format an import error with the correct pip install command."""
    pip_extra = _EXTRA_MAP.get(extra_name, extra_name)
    return (
        f"{class_name} requires the '{extra_name}' package.\n\n"
        f"Install with:  pip install checkllm[{pip_extra}]\n"
        f"Or install everything:  pip install checkllm[all]"
    )
