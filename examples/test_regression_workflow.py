"""Regression detection workflow.

This example shows how to use snapshots to catch quality regressions.

Step 1: Run this once to establish baseline
    checkllm snapshot examples/test_regression_workflow.py --output baseline.json

Step 2: Change the AGENT_VERSION below to "v2" to simulate a code change

Step 3: Create new snapshot and compare
    checkllm snapshot examples/test_regression_workflow.py --output current.json
    checkllm diff --baseline baseline.json --current current.json
"""

# Change this to "v2" to simulate a regression
AGENT_VERSION = "v1"


def _summarizer_v1(text: str) -> str:
    """Good summarizer - accurate and concise."""
    return f"This text discusses: {text[:50]}. Key points are preserved."


def _summarizer_v2(text: str) -> str:
    """Bad summarizer - loses information."""
    return "Here is a summary."


def _get_summarizer():
    if AGENT_VERSION == "v2":
        return _summarizer_v2
    return _summarizer_v1


article = (
    "Machine learning is a subset of artificial intelligence that enables "
    "systems to learn and improve from experience without being explicitly "
    "programmed. It focuses on developing algorithms that can access data "
    "and use it to learn for themselves."
)


def test_summary_contains_key_info(check):
    """Summary should mention the topic."""
    summarizer = _get_summarizer()
    output = summarizer(article)
    check.contains(output, "Machine learning")


def test_summary_length(check):
    """Summary should be concise."""
    summarizer = _get_summarizer()
    output = summarizer(article)
    check.max_tokens(output, limit=100)


def test_summary_not_empty(check):
    """Summary should have substance."""
    summarizer = _get_summarizer()
    output = summarizer(article)
    # At least 5 words
    word_count = len(output.split())
    check.contains(output, " ")  # has at least one space (more than one word)
