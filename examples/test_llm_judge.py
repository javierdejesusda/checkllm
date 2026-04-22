"""LLM-as-judge examples. Requires OPENAI_API_KEY.

Run with: pytest examples/test_llm_judge.py -v
"""

import pytest


@pytest.mark.llm
def test_hallucination_check(check):
    """Verify output is grounded in the provided context."""
    context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    output = "The Eiffel Tower, located in Paris, was constructed in 1889 for the World's Fair."

    check.hallucination(output, context=context)


@pytest.mark.llm
def test_relevance_check(check):
    """Verify output answers the question."""
    output = "Python is a high-level programming language known for its readability."
    check.relevance(output, query="What is Python?")


@pytest.mark.llm
def test_toxicity_check(check):
    """Verify output is safe and appropriate."""
    output = "I'd be happy to help you with that question!"
    check.toxicity(output)


@pytest.mark.llm
def test_rubric_check(check):
    """Verify output meets custom quality criteria."""
    output = "Machine learning is a branch of AI that learns from data."
    check.rubric(
        output,
        criteria="accurate, concise (under 20 words), mentions AI and data",
    )


@pytest.mark.llm
def test_combined_checks(check):
    """Combine deterministic and LLM checks in one test."""
    context = "Python was created by Guido van Rossum and released in 1991."
    output = "Python, created by Guido van Rossum in 1991, is a popular programming language."

    # Fast, free checks first
    check.contains(output, "Python")
    check.contains(output, "Guido")
    check.max_tokens(output, limit=50)

    # Then LLM-as-judge
    check.hallucination(output, context=context, threshold=0.85)
    check.relevance(output, query="Tell me about Python's history")
