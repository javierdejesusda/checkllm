# Quickstart

Get checkllm running in under 5 minutes.

## Install

```bash
pip install checkllm
```

## Initialize

```bash
checkllm init --use-case rag
```

This creates a tailored test file, detects your API keys, and configures `pyproject.toml`.

## Write Your First Test

```python
def test_output_quality(check):
    output = my_llm("What is Python?")

    # Free, instant checks (no API key needed)
    check.contains(output, "programming language")
    check.max_tokens(output, limit=200)
    check.no_pii(output)

    # LLM-as-judge checks (auto-detects your API key)
    check.hallucination(output, context="Python is a programming language.")
    check.relevance(output, query="What is Python?")
```

## Run

```bash
pytest tests/ -v
```

## Estimate Costs Before Running

```bash
checkllm estimate tests/
```

## Track Regressions

```bash
checkllm snapshot tests/ --output baseline.json
# ... make changes ...
checkllm snapshot tests/ --output current.json
checkllm diff --baseline baseline.json --current current.json
```

## Next Steps

- [All 33 deterministic checks](guides/deterministic-checks.md)
- [All 24 LLM metrics](guides/llm-metrics.md)
- [Production guardrails](guides/guardrails.md)
- [CI/CD setup](guides/ci-cd.md)
