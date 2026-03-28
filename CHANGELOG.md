# Changelog

## 0.2.0 (2026-03-28)

### Features

- **Judge response caching**: SQLite-backed content-addressable cache for LLM judge responses. Cache hits return instantly at zero cost. Configurable TTL (default 7 days). CLI commands: `checkllm cache --stats`, `checkllm cache --clear`. Disable with `--no-cache` or `CHECKLLM_CACHE_ENABLED=false`.
- **Parallel judge execution**: Async judge calls are rate-limited via configurable `asyncio.Semaphore` (default 10 concurrent). Use `asyncio.gather` with async check methods for parallel evaluation.
- **Cost budgets**: Set a max USD spend per run with `--budget`, `CHECKLLM_BUDGET`, or `budget` in pyproject.toml. When exceeded, remaining judge calls are skipped with a clear warning.
- **Historical run tracking**: Every test run is recorded in `.checkllm/history.db` with timestamp, git commit, label, scores, and costs. CLI: `checkllm history`, `--run ID`, `--compare ID1,ID2`, `--trend test::metric`.
- **Enhanced run comparison**: Side-by-side diff of two historical runs showing per-test score deltas with color-coded improvements/regressions.
- **CSV and JSON dataset support**: `load_dataset()` and `@dataset()` now auto-detect `.csv`, `.json`, `.yaml`, `.yml` files. CSV extra columns go into `case.metadata`.
- **Customizable judge prompts**: Override the system prompt for any built-in metric via `system_prompt=` kwarg on `check.hallucination()`, `check.relevance()`, `check.toxicity()`, `check.rubric()`, and their async variants.
- **Structured logging**: Python `logging` module integration. Set `CHECKLLM_LOG_LEVEL=DEBUG` or configure in pyproject.toml. Logs cache hits/misses, costs, budget warnings.

### Configuration

New `[tool.checkllm]` options: `cache_enabled`, `cache_dir`, `cache_ttl_seconds`, `max_concurrency`, `budget`, `log_level`.

New environment variables: `CHECKLLM_CACHE_ENABLED`, `CHECKLLM_CACHE_DIR`, `CHECKLLM_CACHE_TTL`, `CHECKLLM_MAX_CONCURRENCY`, `CHECKLLM_BUDGET`, `CHECKLLM_LOG_LEVEL`.

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
