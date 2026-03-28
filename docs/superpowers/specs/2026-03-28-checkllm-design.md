# checkllm — Design Specification

> A Python library for testing LLM-powered applications with the same rigor as traditional software.

**Date:** 2026-03-28
**Status:** Draft
**Language:** Python
**Package name:** `checkllm`

---

## 1. Problem Statement

Most teams shipping LLM features in 2026 test them less rigorously than their login forms. Only 52% of agent developers run offline evaluations; only 37% run online evaluations. The eval landscape is fragmented across DeepEval, RAGAS, promptfoo, and Braintrust with no unified standard.

Key pain points:

- **No pytest-native workflow.** Developers want LLM tests alongside their regular tests, not in a separate tool.
- **No regression detection.** Prompt changes and model upgrades silently degrade quality. No tool catches this with statistical rigor.
- **No plugin ecosystem.** Evaluation needs are domain-specific but no tool offers community-extensible metrics.
- **Fragmented landscape.** Each tool covers a slice; none combines eval metrics + CI/CD integration + regression detection + datasets + reporting.

## 2. Vision

checkllm makes LLM testing feel as natural as pytest makes unit testing. It ships as a **pytest plugin** for CI/CD integration and a **standalone CLI** for development-time evaluation. Developers go from `pip install checkllm` to a passing test in under 5 minutes.

## 3. Core API

### 3.1 pytest Usage (CI/CD)

The `check` fixture collects all check results during a test and raises `CheckFailedError` at test teardown if any check failed. This means a single test can run multiple checks and report all failures, not just the first one.

The `@dataset` decorator generates pytest-parametrized test cases — one test run per case in the dataset. Each case appears as a separate item in pytest output.

```python
# test_my_agent.py
from checkllm import dataset

def test_summarizer_no_hallucination(check):
    """check is a pytest fixture provided by the plugin."""
    result = my_summarizer(source_text)
    check.hallucination(result, context=source_text)

def test_summarizer_quality(check):
    result = my_summarizer(source_text)
    check.relevance(result, query="summarize the article")
    check.rubric(result, criteria="concise, under 3 sentences", threshold=0.8)

@dataset("tests/datasets/summarizer_cases.yaml")
def test_summarizer_dataset(check, case):
    """Runs once per case in the dataset file."""
    result = my_summarizer(case.input)
    check.hallucination(result, context=case.input)
    check.rubric(result, criteria=case.expected_criteria)
```

### 3.2 CLI Usage (development-time)

```bash
# Run all checks with rich terminal output
checkllm run tests/

# Evaluate a single prompt against a dataset
checkllm eval --prompt "Summarize: {text}" --dataset cases.yaml

# Compare two prompt versions side-by-side
checkllm diff --baseline v1.yaml --candidate v2.yaml --dataset cases.yaml

# Generate an HTML report
checkllm report tests/ --output report.html

# Snapshot current results as the regression baseline
checkllm snapshot tests/
```

### 3.3 Configuration

All configuration lives in `pyproject.toml`:

```toml
[tool.checkllm]
judge_model = "gpt-4o"
default_threshold = 0.8
runs_per_test = 1
snapshot_dir = ".checkllm/snapshots"
```

## 4. Built-in Metrics

### 4.1 LLM-as-Judge Metrics (v1)

| Metric | Signature | What it checks |
|--------|-----------|----------------|
| Hallucination | `check.hallucination(output, context)` | Output is grounded in provided context |
| Relevance | `check.relevance(output, query)` | Output actually answers the question |
| Toxicity | `check.toxicity(output)` | No harmful, offensive, or inappropriate content |
| Rubric | `check.rubric(output, criteria, threshold)` | Meets user-defined quality criteria in plain English |

Each metric returns a `CheckResult`:

```python
@dataclass
class CheckResult:
    passed: bool
    score: float          # 0.0 to 1.0
    reasoning: str        # LLM judge's explanation
    cost: float           # USD spent on this evaluation
    latency_ms: int       # Time taken for the eval
    metric_name: str      # e.g., "hallucination"
```

### 4.2 Deterministic Checks (no LLM cost)

These run locally with zero API calls:

```python
check.contains(output, "expected substring")
check.not_contains(output, "forbidden term")
check.max_tokens(output, limit=500)
check.latency(result_meta, max_ms=2000)
check.cost(result_meta, max_usd=0.05)
check.json_schema(output, schema=MyPydanticModel)
check.regex(output, pattern=r"\d{3}-\d{4}")
```

### 4.3 Plugin System

Community-extensible via a decorator-based registration API:

```python
from checkllm import metric, CheckResult

@metric("legal_compliance")
def check_legal_compliance(output: str, context: str, **kwargs) -> CheckResult:
    """Custom metric, available as check.legal_compliance(...)"""
    # evaluation logic
    return CheckResult(passed=True, score=0.95, reasoning="No compliance issues")
```

Plugins are discoverable via the `checkllm.metrics` setuptools entry point:

```toml
# In the plugin's pyproject.toml
[project.entry-points."checkllm.metrics"]
bias = "checkllm_bias:register"
```

This means `pip install checkllm-bias` auto-registers `check.bias()` with no configuration. Same pattern pytest uses for its plugin ecosystem.

## 5. Regression Detection

The killer feature. checkllm treats LLM outputs like snapshot tests.

### 5.1 Workflow

```bash
# 1. Establish a baseline after tests pass
checkllm snapshot tests/

# 2. After a prompt change or model upgrade, compare
checkllm run tests/ --compare

# Output:
# test_summarizer_quality: REGRESSION DETECTED
#   baseline score: 0.92 (avg over 10 runs)
#   current score:  0.71 (avg over 10 runs)
#   delta: -0.21 (p=0.003) -- statistically significant
#   3/10 runs now fail hallucination check (was 0/10)
```

### 5.2 Statistical Rigor

When `runs_per_test > 1`, checkllm runs each test N times and reports:

- **Mean score** and standard deviation
- **Confidence intervals** (95% by default)
- **p-value** via Welch's t-test for regression detection
- **Pass rate** (e.g., 7/10 vs 10/10 baseline)

This addresses the fundamental challenge that LLM outputs are non-deterministic. A single test run is insufficient to detect regression.

### 5.3 CI Integration

```yaml
# GitHub Actions example
- name: Run LLM tests
  run: |
    checkllm run tests/ --compare --fail-on-regression --junit-xml results.xml
```

`--fail-on-regression` exits with code 1 if any metric shows a statistically significant drop (p < 0.05 by default), blocking the PR.

### 5.4 Snapshot Storage

Snapshots are JSON files stored in `.checkllm/snapshots/` (configurable). Each snapshot records:

- Test name and parameters
- Per-metric scores across all runs
- Model and prompt version used
- Timestamp

Snapshots are designed to be committed to git for full traceability.

## 6. Datasets

### 6.1 YAML Datasets

```yaml
# tests/datasets/summarizer_cases.yaml
- input: "Long article about climate change..."
  query: "Summarize this article"
  criteria: "mentions key statistics, under 3 sentences"

- input: "Technical documentation for REST API..."
  query: "Summarize the authentication section"
  criteria: "covers OAuth flow, mentions token expiry"
```

### 6.2 Python Datasets

```python
from checkllm import dataset, Case

@dataset
def financial_cases():
    """Generate test cases programmatically."""
    for ticker in ["AAPL", "GOOG", "MSFT"]:
        data = fetch_latest_financials(ticker)
        yield Case(
            input=data.report,
            query=f"Summarize {ticker} Q4 earnings",
            criteria="mentions revenue, profit, and guidance",
            metadata={"ticker": ticker, "quarter": "Q4"},
        )
```

### 6.3 Dataset Versioning

Datasets are plain files tracked in git. Combined with snapshot baselines, this provides full traceability: which test cases, against which prompt, with which model, produced which results.

## 7. Reporting

### 7.1 CLI Output (development)

Rich terminal output via the `rich` library with color-coded pass/fail, scores, and judge reasoning. Designed for fast iteration.

### 7.2 JUnit XML (CI/CD)

Standard JUnit XML format. Works with GitHub Actions, GitLab CI, Jenkins, CircleCI. Each check maps to a test case in the XML output.

### 7.3 HTML Report (sharing)

```bash
checkllm report tests/ --output report.html
```

Self-contained single HTML file containing:

- Overall pass/fail summary
- Per-test breakdown with scores and reasoning
- Regression trends over time (if snapshots exist)
- Cost summary (total tokens, total USD spent on evaluation)

Designed for non-engineers. Product managers and stakeholders can review LLM quality without reading code.

## 8. Architecture

```
checkllm/
├── core/
│   ├── runner.py            # Test execution engine
│   ├── check.py             # Check fixture, CheckResult model
│   ├── metrics/             # Built-in metric implementations
│   │   ├── __init__.py
│   │   ├── hallucination.py
│   │   ├── relevance.py
│   │   ├── toxicity.py
│   │   └── rubric.py
│   ├── deterministic.py     # Free checks (contains, regex, schema, etc.)
│   └── judge.py             # LLM-as-judge engine (OpenAI)
├── datasets/
│   ├── loader.py            # YAML + Python dataset loading
│   └── case.py              # Case model
├── regression/
│   ├── snapshot.py          # Baseline snapshot management
│   ├── compare.py           # Statistical comparison engine
│   └── stats.py             # p-values, confidence intervals, Welch's t-test
├── reporting/
│   ├── cli.py               # Rich terminal output
│   ├── junit.py             # JUnit XML generation
│   └── html.py              # HTML report generation
├── plugin.py                # Plugin registry + entry point discovery
├── pytest_plugin.py         # pytest integration (fixtures, hooks, markers)
├── cli.py                   # Typer-based CLI entry point
└── config.py                # pyproject.toml config loading via pydantic
```

## 9. Dependencies

| Dependency | Purpose | Required |
|-----------|---------|----------|
| `openai` | LLM-as-judge API calls | Yes |
| `pydantic` | Data models, config validation, schema checks | Yes |
| `typer` | CLI framework | Yes |
| `rich` | Terminal output formatting | Yes |
| `scipy` | Statistical tests for regression detection | Yes |
| `jinja2` | HTML report templating | Yes |
| `pyyaml` | YAML dataset loading | Yes |
| `tiktoken` | Token counting for deterministic checks | Yes |

No heavy frameworks. Minimal dependency tree. Fast install.

## 10. Competitive Positioning

| Feature | checkllm | DeepEval | promptfoo | RAGAS |
|---------|----------|----------|-----------|-------|
| pytest-native | Yes | Yes | No (YAML) | No |
| Standalone CLI | Yes | No | Yes | No |
| Regression snapshots | Statistical | No | Basic | No |
| Plugin system | Entry-point based | No | Limited | No |
| Deterministic checks | Built-in | No | Some | No |
| HTML reports | Yes, free | Dashboard (paid) | Yes | No |
| Config in pyproject.toml | Yes | No | No | No |
| Zero config startup | Yes | Moderate | YAML required | Code required |

The moat is: regression detection with statistical rigor + pytest plugin + community plugin ecosystem. No existing tool combines all three.

## 11. LLM-as-Judge Provider

v1 ships with OpenAI as the sole judge provider. The judge interface is abstracted behind a `JudgeBackend` protocol:

```python
class JudgeBackend(Protocol):
    async def evaluate(self, prompt: str) -> JudgeResponse: ...
```

This allows adding Anthropic, open-source (Ollama), and other providers in v0.2+ without breaking changes. The abstraction exists from day one but only the OpenAI implementation ships.

## 12. Scope Boundaries

**In scope for v1:**
- pytest plugin with `check` fixture
- CLI with `run`, `eval`, `diff`, `snapshot`, `report` commands
- 4 LLM-as-judge metrics (hallucination, relevance, toxicity, rubric)
- 7 deterministic checks
- Plugin system with entry-point discovery
- YAML and Python datasets
- Regression detection with statistical comparison
- CLI, JUnit XML, and HTML reporting
- OpenAI judge backend
- pyproject.toml configuration

**Out of scope for v1 (future):**
- Additional judge providers (Anthropic, Ollama)
- Web dashboard / hosted service
- Real-time production monitoring (online evaluation)
- Prompt versioning and management
- Multi-turn conversation evaluation
- Visual regression testing for UI agents
- Built-in caching of judge responses
