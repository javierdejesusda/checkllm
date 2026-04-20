# Deployment Guide

This guide covers deploying checkllm in production environments, including Docker
containers, multi-GPU judge scaling, and operational hardening.

## Prerequisites

- Python 3.10+ installed
- At least one API key (OpenAI, Anthropic, or Gemini) **or** a local model server
  (Ollama, vLLM)
- Docker 24.0+ (for containerised deployments)
- 2 GB RAM minimum; 8 GB+ recommended when running local judge models

---

## Docker: Single Container

### Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock ./

RUN pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir "checkllm[all]"

COPY . .

CMD ["pytest", "tests/", "-v"]
```

### Build and run

```bash
docker build -t myapp-llm-tests .
docker run \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e CHECKLLM_BUDGET=10.0 \
  myapp-llm-tests
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | one required | — | OpenAI judge key |
| `ANTHROPIC_API_KEY` | one required | — | Anthropic judge key |
| `GEMINI_API_KEY` | one required | — | Gemini judge key |
| `CHECKLLM_BUDGET` | No | unlimited | Max spend per run (USD) |
| `CHECKLLM_JUDGE_MODEL` | No | `gpt-4o` | Default judge model |
| `CHECKLLM_CACHE_ENABLED` | No | `true` | Cache judge responses |
| `CHECKLLM_LOG_LEVEL` | No | `WARNING` | Log verbosity |
| `CHECKLLM_PROFILE` | No | — | Activate a named profile |

---

## Docker Compose: Local Dev with Ollama

Run checkllm alongside a local Ollama instance — zero API cost.

### `docker-compose.yml`

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

  checkllm:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      CHECKLLM_JUDGE_MODEL: "ollama/llama3.2"
      OLLAMA_BASE_URL: "http://ollama:11434"
    command: >
      sh -c "
        curl -s http://ollama:11434/api/pull -d '{\"model\":\"llama3.2\"}' &&
        pytest tests/ -v -m 'not llm'
      "

volumes:
  ollama_data:
```

### Run

```bash
docker compose up --build
```

---

## Multi-GPU Judge Scaling with vLLM

For high-throughput pipelines, run a vLLM server as a local OpenAI-compatible judge
backend across multiple GPUs.

### Start the vLLM server

```bash
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \  # set to number of GPUs
  --max-model-len 8192
```

### Configure checkllm for vLLM

```toml
# pyproject.toml
[tool.checkllm]
judge_model = "meta-llama/Llama-3.1-70B-Instruct"

[tool.checkllm.judge_config]
base_url = "http://vllm-server:8000/v1"
api_key = "not-required"
max_concurrency = 50
```

Or in code:

```python
from checkllm import OpenAICompatibleJudge

judge = OpenAICompatibleJudge(
    model="meta-llama/Llama-3.1-70B-Instruct",
    base_url="http://vllm-server:8000/v1",
    api_key="not-required",
)
```

### Concurrency tuning

| GPU memory | Model size | Recommended `max_concurrency` |
|------------|-----------|-------------------------------|
| 24 GB (RTX 4090) | 7 B | 32 |
| 2x40 GB (A100) | 13 B | 48 |
| 4x80 GB (H100) | 70 B | 64 |

---

## Production Hardening Checklist

- [ ] Set `CHECKLLM_BUDGET` to prevent runaway costs
- [ ] Enable caching (`CHECKLLM_CACHE_ENABLED=true`) to deduplicate evaluations
- [ ] Pin the judge model version: `judge_model = "gpt-4o-2024-11-20"` (not `gpt-4o`)
- [ ] Tune `max_concurrency` to stay within API rate limits
- [ ] Store API keys in a secrets manager (AWS Secrets Manager, HashiCorp Vault)
- [ ] Add `--fail-on-regression` in CI to block score drops
- [ ] Run `checkllm estimate tests/` before any new CI job to sanity-check spend
- [ ] Use `requirements.lock` for reproducible builds (see [Lockfile docs](./configuration.md))

---

## Kubernetes (Quick Reference)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: checkllm-eval
spec:
  template:
    spec:
      containers:
        - name: checkllm
          image: myapp-llm-tests:latest
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-api-keys
                  key: openai
            - name: CHECKLLM_BUDGET
              value: "10.0"
      restartPolicy: Never
```
