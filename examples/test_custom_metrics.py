"""Custom metrics: extend checkllm with your own domain-specific checks.

Run with: pytest examples/test_custom_metrics.py -v
No API key needed.
"""

import checkllm
from checkllm import CheckResult


# --- Register custom metrics ---


@checkllm.metric("word_count")
def word_count_check(
    output: str, min_words: int = 1, max_words: int = 100, **kwargs
) -> CheckResult:
    """Check that output word count is within a range."""
    count = len(output.split())
    passed = min_words <= count <= max_words
    return CheckResult(
        passed=passed,
        score=1.0 if passed else max(0.0, 1.0 - abs(count - max_words) / max_words),
        reasoning=f"{count} words (range: {min_words}-{max_words})",
        cost=0.0,
        latency_ms=0,
        metric_name="word_count",
    )


@checkllm.metric("no_pii")
def no_pii_check(output: str, **kwargs) -> CheckResult:
    """Check that output doesn't contain PII patterns."""
    import re

    pii_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\d{16}\b", "credit card"),
    ]

    found = []
    for pattern, name in pii_patterns:
        if re.search(pattern, output):
            found.append(name)

    passed = len(found) == 0
    return CheckResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reasoning=f"PII found: {', '.join(found)}" if found else "No PII detected",
        cost=0.0,
        latency_ms=0,
        metric_name="no_pii",
    )


# --- Tests using custom metrics ---


def test_response_length(check):
    """Verify response is concise."""
    output = "Python is a versatile programming language used for web development, data science, and automation."
    check.run_metric("word_count", output=output, min_words=5, max_words=30)


def test_no_pii_leak(check):
    """Verify agent doesn't leak personal information."""
    safe_output = "The user's account has been updated successfully."
    check.run_metric("no_pii", output=safe_output)


def test_pii_detected(check):
    """This should fail because the output contains an email."""
    leaky_output = "Contact john.doe@example.com for support."
    check.run_metric("no_pii", output=leaky_output)
    # Note: this test will FAIL because PII is detected - that's intentional
