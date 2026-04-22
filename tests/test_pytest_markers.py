"""Tests for checkllm's pytest marker registration and auto-detection."""

from __future__ import annotations


def test_namespaced_markers_registered(pytester):
    """Each ``checkllm_*`` marker is visible to ``pytest --markers``."""
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest("--markers")
    output = result.stdout.str()
    for marker in (
        "checkllm_rag",
        "checkllm_deterministic",
        "checkllm_llm",
        "checkllm_redteam",
        "checkllm_multimodal",
        "checkllm_slow",
        "checkllm_expensive",
    ):
        assert marker in output, f"missing marker {marker!r} in --markers output"


def test_deterministic_selection(pytester):
    """``-m checkllm_deterministic`` selects only deterministic tests."""
    pytester.makepyfile(
        """
        def test_contains_only(check):
            check.contains('hello world', 'hello')

        def test_llm(check):
            # Auto-detected as LLM; should be excluded by the selector.
            try:
                check.hallucination('x', context='y')
            except Exception:
                pass
        """
    )
    result = pytester.runpytest("-m", "checkllm_deterministic", "-v")
    result.assert_outcomes(passed=1)


def test_llm_auto_marker(pytester):
    """Tests calling an LLM judge get ``checkllm_llm``."""
    pytester.makepyfile(
        """
        def test_llm(check):
            try:
                check.faithfulness('answer', context='ctx')
            except Exception:
                pass

        def test_deterministic(check):
            check.contains('a', 'a')
        """
    )
    result = pytester.runpytest("-m", "checkllm_llm", "-v")
    # Only the faithfulness test should match; CheckJudgeError is
    # converted to a skip by the plugin when no API key is present.
    assert "test_llm" in result.stdout.str()
    assert "test_deterministic" not in result.stdout.str()


def test_user_marker_is_not_overridden(pytester):
    """User-supplied ``checkllm_*`` markers take precedence over auto-detection."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.checkllm_slow
        def test_custom(check):
            check.contains('a', 'a')
        """
    )
    # The test should only be selected when we filter by the user-set marker.
    result = pytester.runpytest("-m", "checkllm_slow", "-v")
    result.assert_outcomes(passed=1)
    # And it should NOT be auto-tagged deterministic.
    result = pytester.runpytest("-m", "checkllm_deterministic", "-v")
    result.assert_outcomes()


def test_rag_marker_applied_for_rag_metrics(pytester):
    """``check.faithfulness`` tags the test with both llm and rag."""
    pytester.makepyfile(
        """
        def test_rag(check):
            try:
                check.faithfulness('answer', context='ctx')
            except Exception:
                pass
        """
    )
    result = pytester.runpytest("-m", "checkllm_rag", "-v")
    assert "test_rag" in result.stdout.str()


def test_auto_marker_helper_direct():
    """``apply_checkllm_markers`` returns marker names for inspection."""
    from checkllm.pytest_plugin import _metric_names_in_source

    src = (
        "def test_x(check):\n"
        "    check.contains('a', 'a')\n"
        "    check.hallucination('a', context='b')\n"
    )
    metrics = _metric_names_in_source(src)
    assert "contains" in metrics
    assert "hallucination" in metrics


def test_auto_marker_handles_async_prefix():
    """Async variants such as ``ahallucination`` map back to the base metric."""
    from checkllm.pytest_plugin import _metric_names_in_source

    src = "async def test_x(check):\n    await check.ahallucination('a', context='b')\n"
    metrics = _metric_names_in_source(src)
    assert "hallucination" in metrics


def test_auto_marker_silent_on_source_failure(monkeypatch):
    """``apply_checkllm_markers`` swallows inspect failures."""
    import inspect

    from checkllm.pytest_plugin import apply_checkllm_markers

    class _FakeItem:
        function = lambda: None  # noqa: E731

        def iter_markers(self):
            return iter(())

        def add_marker(self, _marker):
            raise AssertionError("should not be called")

    def _raise(*_a, **_kw):
        raise OSError("source unavailable")

    monkeypatch.setattr(inspect, "getsource", _raise)

    # No exception propagates; an empty iterable is returned.
    added = list(apply_checkllm_markers(_FakeItem()))
    assert added == []
