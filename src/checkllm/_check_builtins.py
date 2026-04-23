"""Register the built-in deterministic checks into :data:`CHECK_REGISTRY`.

This module is imported for its side effects. It walks
:class:`checkllm.deterministic.DeterministicChecks`, grabs every public
method whose name appears in the curated list below, and registers a
thin wrapper in :data:`CHECK_REGISTRY` with sensible tags.

Keeping registration out of ``deterministic.py`` avoids module-level
coupling and keeps the registry optional — users who never touch
``check_registry`` pay nothing at import time beyond this one pass.

Backward compatibility: the wrappers delegate to the same singleton
``DeterministicChecks`` instance, so behavior is byte-for-byte identical
to calling the method directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from checkllm.check_registry import CHECK_REGISTRY, CheckCallable
from checkllm.deterministic import DeterministicChecks
from checkllm.models import CheckResult

# One shared instance is fine: every registered check is stateless.
_SHARED = DeterministicChecks()


def _wrap(method_name: str) -> CheckCallable:
    method = getattr(_SHARED, method_name)

    def wrapper(*args: Any, **kwargs: Any) -> CheckResult:
        result: CheckResult = method(*args, **kwargs)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = method.__doc__
    wrapper.__qualname__ = f"DeterministicChecks.{method_name}"
    return wrapper


# Grouping by tag for discovery. The tag vocabulary is intentionally
# small; new checks can add more tags without migration work.
_BUILTIN_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Substring / prefix / suffix.
    ("contains", ("string", "substring")),
    ("not_contains", ("string", "substring")),
    ("icontains", ("string", "substring", "case-insensitive")),
    ("icontains_any", ("string", "substring", "case-insensitive")),
    ("icontains_all", ("string", "substring", "case-insensitive")),
    ("starts_with", ("string",)),
    ("ends_with", ("string",)),
    ("exact_match", ("string", "equality")),
    ("exact_match_strict", ("string", "equality")),
    ("regex", ("string", "pattern")),
    # Set-theoretic aggregates.
    ("all_of", ("composite", "substring")),
    ("any_of", ("composite", "substring")),
    ("none_of", ("composite", "substring")),
    # Length / counting.
    ("max_tokens", ("length",)),
    ("min_tokens", ("length",)),
    ("word_count", ("length",)),
    ("char_count", ("length",)),
    ("sentence_count", ("length",)),
    # Cost and latency.
    ("latency", ("perf",)),
    ("cost", ("perf",)),
    ("latency_check", ("perf",)),
    ("cost_check", ("perf",)),
    # Structure / format validation.
    ("json_schema", ("format", "json")),
    ("is_json", ("format", "json")),
    ("is_valid_python", ("format", "code")),
    ("is_valid_sql", ("format", "code", "sql")),
    ("is_valid_yaml", ("format", "yaml")),
    ("is_yaml", ("format", "yaml")),
    ("is_html", ("format", "html")),
    ("contains_html", ("format", "html")),
    ("is_xml", ("format", "xml")),
    ("contains_xml", ("format", "xml")),
    ("is_valid_url", ("format", "url")),
    ("is_url", ("format", "url")),
    ("has_url", ("format", "url")),
    ("is_valid_markdown", ("format", "markdown")),
    ("json_field", ("format", "json")),
    ("has_structure", ("format", "structure")),
    # Similarity / reference-based.
    ("similarity", ("similarity", "string")),
    ("levenshtein", ("similarity", "string")),
    ("string_distance", ("similarity", "string")),
    ("semantic_similarity", ("similarity", "tfidf")),
    ("bleu", ("similarity", "reference", "mt")),
    ("rouge_l", ("similarity", "reference", "summarization")),
    ("meteor", ("similarity", "reference", "mt")),
    ("gleu", ("similarity", "reference", "mt")),
    ("chrf", ("similarity", "reference", "mt")),
    # Quality heuristics.
    ("readability", ("quality", "readability")),
    ("perplexity_check", ("quality", "readability")),
    ("no_repetition", ("quality",)),
    ("has_citations", ("quality", "citations")),
    ("is_refusal", ("safety", "behavior")),
    # Privacy.
    ("no_pii", ("safety", "privacy")),
    # Language & comparators.
    ("language", ("lang",)),
    ("greater_than", ("numeric",)),
    ("less_than", ("numeric",)),
    ("between", ("numeric",)),
)


def _register_all() -> None:
    """Populate :data:`CHECK_REGISTRY` with the built-in deterministic checks.

    Idempotent: re-imports won't raise, they will silently overwrite.
    """
    for method_name, tags in _BUILTIN_CHECKS:
        if not hasattr(_SHARED, method_name):
            # Defensive: if someone removes a method upstream, skip
            # registration rather than crashing module import.
            continue
        wrapper = _wrap(method_name)
        CHECK_REGISTRY.register(
            method_name,
            wrapper,
            tags=tags,
            source="builtin",
            overwrite=True,
        )


_register_all()


def registered_deterministic_checks() -> list[str]:
    """Return the names registered by this module (for introspection/tests)."""
    return [name for name, _ in _BUILTIN_CHECKS if name in CHECK_REGISTRY]


# Public re-export so ``from checkllm._check_builtins import registered_deterministic_checks``
# works without touching private symbols.
__all__: list[str] = ["registered_deterministic_checks"]


def _unused_type_anchor() -> Callable[..., CheckResult]:  # pragma: no cover
    """Keep ``CheckCallable`` referenced for static analysis."""
    return lambda *a, **k: CheckResult(
        passed=True,
        score=1.0,
        reasoning="",
        cost=0.0,
        latency_ms=0,
        metric_name="noop",
    )


_ = CheckCallable  # pragma: no cover - suppresses unused-import mypy warnings
