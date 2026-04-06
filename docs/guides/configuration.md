# Configuration

## pyproject.toml

```toml
[tool.checkllm]
judge_backend = "auto"           # auto, openai, anthropic, gemini, azure, ollama, litellm
judge_model = "gpt-4o"
default_threshold = 0.8
runs_per_test = 1
engine = "auto"                  # async, thread, process, hybrid, auto
max_concurrency = 10
budget = 10.00                   # Max USD per run (optional)
cache_enabled = true
cache_ttl_seconds = 604800       # 7 days
log_level = "WARNING"
```

## Profiles

```toml
[tool.checkllm.profiles.dev]
judge_model = "gpt-4o-mini"
budget = 1.0
log_level = "DEBUG"

[tool.checkllm.profiles.ci]
cache_enabled = false
budget = 10.0
engine = "async"

[tool.checkllm.profiles.prod]
judge_model = "gpt-4o"
default_threshold = 0.9
max_concurrency = 20
```

Activate: `CHECKLLM_PROFILE=ci checkllm run tests/`

## Environment Variables

All settings support env var overrides:

| Variable | Config Key |
|----------|-----------|
| `CHECKLLM_JUDGE_BACKEND` | `judge_backend` |
| `CHECKLLM_JUDGE_MODEL` | `judge_model` |
| `CHECKLLM_DEFAULT_THRESHOLD` | `default_threshold` |
| `CHECKLLM_ENGINE` | `engine` |
| `CHECKLLM_BUDGET` | `budget` |
| `CHECKLLM_CACHE_ENABLED` | `cache_enabled` |
| `CHECKLLM_MAX_CONCURRENCY` | `max_concurrency` |
| `CHECKLLM_LOG_LEVEL` | `log_level` |
| `CHECKLLM_PROFILE` | Active profile |
