"""Dataset-driven testing: run the same checks across many inputs.

Run with: pytest examples/test_dataset_driven.py -v
No API key needed.
"""
from checkllm import Case, dataset


# --- YAML dataset ---

@dataset("examples/qa_dataset.yaml")
def test_qa_from_yaml(check, case):
    """Test a Q&A agent across cases loaded from YAML."""
    # In real usage, you'd call your agent here:
    # output = my_agent(case.input)

    # For this example, simulate the agent output
    output = _fake_qa_agent(case.input)

    # Check the output
    if case.expected:
        check.contains(output, case.expected)
    check.max_tokens(output, limit=200)


# --- Python generator dataset ---

def regression_cases():
    """Generate test cases programmatically."""
    yield Case(
        input="What is the capital of France?",
        expected="Paris",
        criteria="mentions Paris, concise",
    )
    yield Case(
        input="What is 2 + 2?",
        expected="4",
        criteria="correct answer",
    )
    yield Case(
        input="Who wrote Romeo and Juliet?",
        expected="Shakespeare",
        criteria="mentions Shakespeare",
    )


@dataset(regression_cases)
def test_qa_from_generator(check, case):
    """Test a Q&A agent across programmatic cases."""
    output = _fake_qa_agent(case.input)

    if case.expected:
        check.contains(output, case.expected)


# --- Helper ---

def _fake_qa_agent(question: str) -> str:
    """Simulate an LLM agent for demonstration purposes."""
    answers = {
        "What is Python?": "Python is a high-level programming language created by Guido van Rossum.",
        "What is the capital of France?": "The capital of France is Paris.",
        "What is 2 + 2?": "The answer is 4.",
        "Who wrote Romeo and Juliet?": "Romeo and Juliet was written by William Shakespeare.",
        "Explain machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
    }
    return answers.get(question, f"I don't have an answer for: {question}")
