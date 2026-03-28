# Changelog

## 0.1.0 (2026-03-28)

Initial release.

### Features

- **pytest plugin** with `check` fixture for LLM testing in pytest
- **Deterministic checks**: `contains`, `not_contains`, `regex`, `json_schema`, `max_tokens`, `latency`, `cost`
- **LLM-as-judge metrics**: `hallucination`, `relevance`, `toxicity`, `rubric`
- **Custom metrics** via `@metric` decorator and plugin entry points
- **Dataset system**: YAML loading, generator functions, `@dataset` decorator for parametrized tests
- **Regression detection**: Welch's t-test with configurable p-value threshold
- **Snapshot system**: save/load/compare test result baselines
- **Reporting**: Rich terminal output, HTML reports, JUnit XML
- **CLI**: `checkllm run`, `snapshot`, `report`, `eval`, `diff`, `init`
- **Multiple judge backends**: OpenAI and Anthropic
- **Retry logic** with exponential backoff for transient API failures
- **Cost tracking** from OpenAI/Anthropic token usage
- **Configuration** via `pyproject.toml [tool.checkllm]` and environment variables
