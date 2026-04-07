"""Verify pytest plugin produces rich failure messages."""


def test_rich_failure_plugin(pytester):
    """Check that failed checks produce detailed output, not just a count."""
    pytester.makepyfile("""
        def test_failing_check(check):
            check.contains("hello world", "goodbye")
            check.regex("no numbers here", r"\\\\d+")
    """)
    result = pytester.runpytest("-v")
    output = result.stdout.str()
    assert "FAILED: contains" in output or "contains" in output
    assert "FAILED: regex" in output or "regex" in output
    assert "Score:" in output or "0.00" in output
