# checkllm Examples

Run any example with `pytest`:

```bash
# Basic deterministic checks (no API key needed)
pytest examples/test_basic.py -v

# Dataset-driven testing
pytest examples/test_dataset_driven.py -v

# Custom metrics
pytest examples/test_custom_metrics.py -v

# Regression detection workflow
checkllm snapshot examples/test_regression_workflow.py --output baseline.json
# (change AGENT_VERSION to "v2" in the file)
checkllm snapshot examples/test_regression_workflow.py --output current.json
checkllm diff --baseline baseline.json --current current.json
```

## LLM-as-Judge Examples

These require `OPENAI_API_KEY` to be set:

```bash
export OPENAI_API_KEY=sk-...
pytest examples/test_llm_judge.py -v
```
