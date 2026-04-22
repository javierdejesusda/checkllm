# Contributing to checkllm

Thanks for your interest in contributing to checkllm! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/checkllm/checkllm.git
cd checkllm
pip install -e ".[dev]"
pytest tests/
```

## Project Structure

```
src/checkllm/
  check.py              # Core CheckCollector (the `check` fixture)
  deterministic.py      # 33 free, instant checks
  metrics/              # 24 LLM-as-judge metrics
  judge.py              # JudgeBackend protocol + OpenAI/Anthropic
  providers.py          # Gemini, Azure, Ollama, LiteLLM, CustomHTTP
  discovery.py          # Auto-detect judge from environment
  config.py             # Config loading from pyproject.toml
  cli.py                # Typer CLI commands
  pytest_plugin.py      # pytest hook integration
  guardrails.py         # Production runtime validation
  chain.py              # Fluent assertion chaining
  estimator.py          # Cost estimation
  reporting/            # HTML, Markdown, JUnit, JSONL, CSV, GitHub
  regression/           # Snapshot comparison, Welch's t-test
  templates/            # Init scaffolding templates
tests/                  # 1100+ tests
```

## How to Contribute

### Bug fixes

1. Open an issue describing the bug
2. Fork the repo and create a branch: `git checkout -b fix/description`
3. Write a failing test that reproduces the bug
4. Fix the bug
5. Run `pytest tests/` to verify
6. Submit a PR

### New deterministic check

1. Add the check method to `src/checkllm/deterministic.py`
2. Add a wrapper to `src/checkllm/check.py` in the `CheckCollector` class
3. Add to the chain in `src/checkllm/chain.py` (optional)
4. Write tests in `tests/`
5. Update the `list-metrics` command in `cli.py`

### New LLM metric

1. Create `src/checkllm/metrics/your_metric.py` following existing patterns
2. Add a wrapper to `src/checkllm/check.py` using `_cached_judge_check`
3. Import it in `src/checkllm/check.py`
4. Write tests in `tests/test_metrics/`
5. Update the `list-metrics` command in `cli.py`

### Plugin metric (community package)

You can publish metrics as separate packages:

```python
# my_checkllm_plugin/metric.py
from checkllm import metric, CheckResult

@metric("my_custom_metric")
def my_custom_metric(output: str, **kwargs) -> CheckResult:
    return CheckResult(
        passed=True, score=0.9, reasoning="Custom logic",
        cost=0.0, latency_ms=0, metric_name="my_custom_metric",
    )
```

```toml
# pyproject.toml
[project.entry-points."checkllm.metrics"]
my_custom_metric = "my_checkllm_plugin.metric:my_custom_metric"
```

## Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to keep the tree tidy. After
cloning, install the hooks once:

```bash
pip install pre-commit
pre-commit install
```

Run the full suite manually at any time:

```bash
pre-commit run --all-files
```

The hooks cover trailing whitespace, YAML/TOML validity, large-file and
private-key detection, `ruff` lint + format, `mypy` (loose mode),
`bandit` on `src/`, `codespell`, and a local guard that blocks
disallowed author-attribution references from being committed (see
the project style rules for the exact list).

## Code Style

- Type hints on all public APIs
- Docstrings on public classes and functions
- Follow existing patterns in the codebase
- Keep files focused — one responsibility per module
- Async-first for judge backends (sync wrappers use `_run_async`)

## Testing

```bash
# Run all tests (except those needing API keys)
pytest tests/ -k "not llm"

# Run a specific test file
pytest tests/test_check.py -v

# Run with coverage
pytest tests/ --cov=checkllm --cov-report=html
```

Tests requiring API keys are marked with `@pytest.mark.llm` and skipped in CI.

## Commit Messages

Use conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — tests
- `refactor:` — code restructuring

## Questions?

Open an issue or start a discussion on GitHub.
