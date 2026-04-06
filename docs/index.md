# checkllm

**The pytest of LLM testing.**

checkllm lets you test LLM-powered applications with the same rigor as traditional software. Add a `check` parameter to any pytest test and start asserting on LLM outputs — no setup, no boilerplate.

```python
def test_my_llm(check):
    output = my_llm("What is Python?")
    check.contains(output, "programming language")
    check.no_pii(output)
    check.hallucination(output, context="Python is a programming language.")
```

## Key Features

- **33 free checks** — run instantly with zero API calls
- **24 LLM-as-judge metrics** — hallucination, relevance, faithfulness, bias, and more
- **7 judge backends** — OpenAI, Anthropic, Gemini, Azure, Ollama, LiteLLM, custom
- **Production guardrails** — same checks in tests and production
- **Auto-detect judge** — zero config when API keys are set
- **Cost estimation** — know what you'll spend before running
- **Fluent chaining** — `check.that(output).contains("X").has_no_pii()`

## Get Started

```bash
pip install checkllm
checkllm init
```

[Quickstart Guide](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/checkllm/checkllm){ .md-button }
