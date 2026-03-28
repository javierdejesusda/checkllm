"""Basic checkllm usage: deterministic checks on LLM outputs.

Run with: pytest examples/test_basic.py -v
No API key needed.
"""


def test_output_has_required_content(check):
    """Verify that an LLM output contains expected information."""
    # Simulate calling your LLM agent
    output = "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms."

    # Verify the output contains key information
    check.contains(output, "Python")
    check.contains(output, "Guido van Rossum")
    check.contains(output, "1991")

    # Verify nothing unexpected
    check.not_contains(output, "JavaScript")
    check.not_contains(output, "error")


def test_output_length(check):
    """Verify output stays within token limits."""
    output = "Python is a versatile programming language."

    check.max_tokens(output, limit=100)


def test_exact_expected_output(check):
    """Verify exact output for deterministic LLM calls."""
    output = "42"
    check.exact_match(output, "42")


def test_json_structured_output(check):
    """Verify LLM returns valid JSON matching a schema."""
    from pydantic import BaseModel

    class AgentResponse(BaseModel):
        answer: str
        confidence: float
        sources: list[str]

    output = '{"answer": "Paris", "confidence": 0.95, "sources": ["Wikipedia"]}'
    check.json_schema(output, schema=AgentResponse)


def test_output_format(check):
    """Verify output follows expected format patterns."""
    output = "Step 1: Gather requirements\nStep 2: Design solution\nStep 3: Implement"

    check.starts_with(output, "Step 1:")
    check.regex(output, pattern=r"Step \d+:")
    check.not_contains(output, "TODO")


def test_cost_and_latency(check):
    """Verify API calls stay within budget and latency limits."""
    # These would come from your actual API call metrics
    api_cost = 0.003  # $0.003
    response_time = 450  # 450ms

    check.cost(api_cost, max_usd=0.01)
    check.latency(response_time, max_ms=2000)
