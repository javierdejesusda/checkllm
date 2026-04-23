"""Tests for checkllm.check_registry and composition primitives."""

from __future__ import annotations

import pytest

from checkllm.check_registry import (
    CHECK_REGISTRY,
    AllOf,
    AnyOf,
    CheckRegistry,
    Not,
    check,
    run_check,
)
from checkllm.models import CheckResult


def _make(passed: bool, score: float, metric_name: str = "fake") -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        reasoning=f"{metric_name} -> {passed}",
        cost=0.0,
        latency_ms=0,
        metric_name=metric_name,
    )


class TestRegistry:
    def test_fresh_registry_starts_empty(self):
        reg = CheckRegistry()
        assert len(reg) == 0
        assert reg.names() == []

    def test_register_and_lookup(self):
        reg = CheckRegistry()

        def my_check(output: str) -> CheckResult:
            """My docstring first line.

            More.
            """
            return _make(True, 1.0)

        reg.register("my_check", my_check, tags=("demo",))
        assert "my_check" in reg
        spec = reg.get("my_check")
        assert spec.name == "my_check"
        assert spec.tags == ("demo",)
        assert spec.description == "My docstring first line."

    def test_duplicate_registration_rejected(self):
        reg = CheckRegistry()
        reg.register("dup", lambda o: _make(True, 1.0))
        with pytest.raises(ValueError):
            reg.register("dup", lambda o: _make(True, 1.0))

    def test_overwrite_ok_when_asked(self):
        reg = CheckRegistry()
        reg.register("dup", lambda o: _make(True, 1.0))
        reg.register("dup", lambda o: _make(False, 0.0), overwrite=True)
        assert reg.get("dup").func("x").passed is False

    def test_unregister_is_noop_on_missing(self):
        reg = CheckRegistry()
        reg.unregister("never_added")  # does not raise

    def test_filter_by_tag(self):
        reg = CheckRegistry()
        reg.register("a", lambda o: _make(True, 1.0), tags=("safety",))
        reg.register("b", lambda o: _make(True, 1.0), tags=("format",))
        reg.register("c", lambda o: _make(True, 1.0), tags=("safety", "format"))
        safety_specs = reg.filter_by_tag("safety")
        assert {s.name for s in safety_specs} == {"a", "c"}

    def test_lookup_missing_raises(self):
        reg = CheckRegistry()
        with pytest.raises(KeyError):
            reg.get("ghost")


class TestCheckDecorator:
    def teardown_method(self):
        CHECK_REGISTRY.unregister("decorator_scratch")
        CHECK_REGISTRY.unregister("decorator_scratch_renamed")

    def test_decorator_registers_by_function_name(self):
        @check(tags=("demo",))
        def decorator_scratch(output: str) -> CheckResult:
            """Scratch check."""
            return _make(True, 1.0, "decorator_scratch")

        assert "decorator_scratch" in CHECK_REGISTRY
        assert CHECK_REGISTRY.get("decorator_scratch").tags == ("demo",)
        # The decorated function still works directly (no wrapping).
        assert decorator_scratch("x").passed is True

    def test_explicit_name_overrides(self):
        @check("decorator_scratch_renamed", tags=("x",))
        def actual_func(output: str) -> CheckResult:
            return _make(True, 1.0, "renamed")

        assert "decorator_scratch_renamed" in CHECK_REGISTRY
        assert run_check("decorator_scratch_renamed", "x").passed is True
        assert actual_func.__checkllm_check_name__ == "decorator_scratch_renamed"  # type: ignore[attr-defined]


class TestBuiltinsAreRegistered:
    def test_all_deterministic_checks_present(self):
        # Importing the package runs _check_builtins; a representative sample
        # of the 39+ built-ins must be present.
        expected = {
            "contains",
            "not_contains",
            "max_tokens",
            "min_tokens",
            "word_count",
            "char_count",
            "sentence_count",
            "latency",
            "cost",
            "json_schema",
            "regex",
            "exact_match",
            "starts_with",
            "ends_with",
            "similarity",
            "readability",
            "all_of",
            "any_of",
            "none_of",
            "is_json",
            "is_valid_python",
            "no_pii",
            "language",
            "greater_than",
            "less_than",
            "between",
            "bleu",
            "rouge_l",
            "is_valid_sql",
            "icontains",
            "icontains_any",
            "icontains_all",
            "is_html",
            "is_xml",
            "is_refusal",
            "levenshtein",
            "meteor",
            "is_valid_yaml",
            "has_citations",
            "no_repetition",
        }
        missing = expected - set(CHECK_REGISTRY.names())
        assert missing == set(), f"Missing built-in checks: {missing}"

    def test_builtin_check_runs(self):
        result = run_check("contains", "hello world", "hello")
        assert result.passed is True
        assert result.metric_name == "contains"

    def test_builtin_has_tags(self):
        spec = CHECK_REGISTRY.get("no_pii")
        assert "safety" in spec.tags


class TestAllOf:
    def test_all_pass(self):
        pass_check = lambda o: _make(True, 0.8, "p1")  # noqa: E731
        pass2 = lambda o: _make(True, 0.9, "p2")  # noqa: E731
        c = AllOf(pass_check, pass2, name="all_pass")
        result = c("x")
        assert result.passed is True
        assert result.score == pytest.approx(0.8)  # min score
        assert result.metric_name == "all_pass"
        assert "all passed" in result.reasoning

    def test_one_fails_fails_all(self):
        c = AllOf(
            lambda o: _make(True, 1.0, "p"),
            lambda o: _make(False, 0.2, "f"),
        )
        result = c("x")
        assert result.passed is False
        assert "f" in result.reasoning

    def test_works_with_registered_names(self):
        c = AllOf("contains", name="combo")
        # "substring" is present in "hello substring", so contains passes.
        result = c("hello substring", "substring")
        assert result.passed is True

    def test_empty_composite_passes_trivially(self):
        c = AllOf()
        result = c("anything")
        assert result.passed is True  # vacuously true
        assert result.score == 0.0


class TestAnyOf:
    def test_any_passes_ok(self):
        c = AnyOf(
            lambda o: _make(False, 0.2, "a"),
            lambda o: _make(True, 0.6, "b"),
        )
        result = c("x")
        assert result.passed is True
        assert result.score == pytest.approx(0.6)
        assert "b" in result.reasoning

    def test_all_fail(self):
        c = AnyOf(
            lambda o: _make(False, 0.1, "a"),
            lambda o: _make(False, 0.3, "b"),
        )
        result = c("x")
        assert result.passed is False
        assert result.score == pytest.approx(0.3)  # max among fails

    def test_short_circuit_not_required_but_returns_best_reason(self):
        c = AnyOf(
            lambda o: _make(True, 1.0, "win1"),
            lambda o: _make(True, 0.5, "win2"),
        )
        result = c("x")
        assert result.passed is True
        assert "win1" in result.reasoning or "win2" in result.reasoning


class TestNot:
    def test_inverts_pass(self):
        c = Not(lambda o: _make(True, 0.8, "inner"))
        result = c("x")
        assert result.passed is False
        assert result.score == pytest.approx(0.2)

    def test_inverts_fail(self):
        c = Not(lambda o: _make(False, 0.1, "inner"))
        result = c("x")
        assert result.passed is True
        assert result.score == pytest.approx(0.9)

    def test_nested_composition(self):
        # Not(AnyOf(contains_foo, contains_bar))
        contains_foo = lambda o: _make("foo" in o, 1.0 if "foo" in o else 0.0, "foo")  # noqa: E731
        contains_bar = lambda o: _make("bar" in o, 1.0 if "bar" in o else 0.0, "bar")  # noqa: E731
        none_of_them = Not(AnyOf(contains_foo, contains_bar), name="neither")
        assert none_of_them("baz").passed is True
        assert none_of_them("foo and baz").passed is False

    def test_score_clamped_to_unit_interval(self):
        # Even if inner returns an exact 0 or 1, score must stay in [0,1].
        c_zero = Not(lambda o: _make(False, 0.0, "z"))
        c_one = Not(lambda o: _make(True, 1.0, "o"))
        assert c_zero("x").score == 1.0
        assert c_one("x").score == 0.0
