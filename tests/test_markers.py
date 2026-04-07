"""Verify checkllm registers useful pytest markers."""
import pytest


def test_markers_registered(pytester):
    """Check that checkllm markers are registered and usable."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.llm
        def test_llm_check(check):
            pass

        @pytest.mark.deterministic
        def test_deterministic_check(check):
            check.contains("hello", "hello")

        @pytest.mark.expensive
        def test_expensive_check(check):
            pass
    """)
    result = pytester.runpytest("-m", "deterministic", "-v")
    result.assert_outcomes(passed=1)

    result = pytester.runpytest("-m", "not expensive", "-v")
    result.assert_outcomes(passed=2)
