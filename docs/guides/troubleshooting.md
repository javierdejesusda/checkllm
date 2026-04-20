# Troubleshooting Guide

Solutions for the most common checkllm issues.

---

## API Key Issues

### `AuthenticationError: No API key found`

**Cause:** The judge backend cannot locate the required API key.

**Fix:**

```bash
export OPENAI_API_KEY="sk-..."           # OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."    # Anthropic
export GEMINI_API_KEY="AIza..."          # Gemini
```

Verify the key is loaded before running tests:

```bash
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

In GitHub Actions, expose the secret explicitly:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Rate Limiting (429 Errors)

### `RateLimitError: Too many requests`

**Cause:** Too many concurrent requests to the judge API.

**Fix 1 — Reduce concurrency:**

```toml
[tool.checkllm]
max_concurrency = 5   # default is 10; lower for free-tier keys
```

**Fix 2 — Enable caching** to skip duplicate calls:

```toml
[tool.checkllm]
cache_enabled = true
```

**Fix 3 — Use a cheaper model for development:**

```toml
[tool.checkllm.profiles.dev]
judge_model = "gpt-4o-mini"   # ~15x cheaper than gpt-4o
```

---

## Judge Timeout / Hanging Tests

### `TimeoutError` or tests that never finish

**Cause:** The judge API is unreachable or responding very slowly.

**Fix — Add explicit timeouts:**

```python
from checkllm import OpenAIJudge

judge = OpenAIJudge(model="gpt-4o", timeout=30, max_retries=3)
```

**Debug — Test connectivity:**

```bash
checkllm ping --judge openai
# Output: checkllm OpenAI judge reachable (latency: 342 ms)
```

---

## Flaky Tests / Score Variance

### A test passes inconsistently across runs

**Cause:** LLM judges are non-deterministic by default.

**Fix 1 — Set temperature to 0:**

```python
from checkllm import OpenAIJudge

judge = OpenAIJudge(model="gpt-4o", temperature=0)
```

**Fix 2 — Average over multiple runs:**

```toml
[tool.checkllm]
runs_per_test = 3
```

**Fix 3 — Loosen the threshold slightly to account for variance:**

```python
@pytest.mark.llm_check(metric="hallucination", threshold=0.75)  # was 0.80
def test_no_hallucination(response):
    ...
```

---

## Coverage Threshold Not Met

### `FAILED - Required test coverage of 75% not reached`

This is a `pytest-cov` failure, not a checkllm metric failure.

**Fix — Find uncovered lines:**

```bash
pytest --cov=src/checkllm --cov-report=term-missing | grep "Miss"
```

Add tests for any lines listed in the `Miss` column.

---

## Import Errors

### `ModuleNotFoundError: No module named 'openai'`

**Cause:** Optional dependency not installed.

```bash
pip install "checkllm[openai]"      # OpenAI judge
pip install "checkllm[anthropic]"   # Anthropic judge
pip install "checkllm[all]"         # Install everything
```

---

## Async Issues

### `RuntimeError: This event loop is already running`

**Fix — Ensure `asyncio_mode = "auto"` in `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```python
import pytest

@pytest.mark.asyncio
async def test_async_call():
    result = await my_async_llm_call()
    assert result is not None
```

---

## Snapshot File Corruption

### `JSONDecodeError` when running `checkllm diff`

**Cause:** A snapshot was partially written (interrupted run).

**Fix — Delete and re-baseline:**

```bash
rm -rf .checkllm/snapshots/
checkllm snapshot --save
```

---

## Debug Mode

Enable full verbose output for any issue:

```bash
CHECKLLM_LOG_LEVEL=DEBUG pytest tests/ -v -s 2>&1 | tee debug.log
```

Or configure a debug profile:

```toml
[tool.checkllm.profiles.debug]
log_level = "DEBUG"
judge_model = "gpt-4o-mini"
cache_enabled = false
```

Run with:

```bash
CHECKLLM_PROFILE=debug pytest tests/ -v -s
```

---

## Getting Help

- **GitHub Issues:** https://github.com/javierdejesusda/checkllm/issues
- **Discussions:** https://github.com/javierdejesusda/checkllm/discussions
- **Security issues:** see [SECURITY.md](https://github.com/javierdejesusda/checkllm/blob/main/SECURITY.md)
