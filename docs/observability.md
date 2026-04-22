# Observability

checkllm ships with first-class exporters for popular tracing and
metrics backends. All of them plug into the same
`checkllm.tracing.Tracer` API used internally so adoption is additive:
drop in a tracer and every evaluation span is mirrored to your backend.

| Backend        | Class / function                             | Extra          |
|----------------|----------------------------------------------|----------------|
| LangSmith      | `LangSmithTracer`                            | `langsmith`    |
| LangFuse       | `LangFuseTracer`                             | `langfuse`     |
| Datadog (OTLP) | `DatadogTracer`                              | `datadog`      |
| Prometheus     | `PrometheusExporter`                         | `prometheus`   |
| Cloud sync     | `checkllm.integrations.cloud_sync.push_to_remote` | (stdlib)  |

Install everything at once with:

```bash
pip install "checkllm[all]"
```

Or pick a single backend:

```bash
pip install "checkllm[langsmith]"
pip install "checkllm[langfuse]"
pip install "checkllm[datadog]"
pip install "checkllm[prometheus]"
```

## Factory

Every tracer is reachable by name through `get_tracer`:

```python
from checkllm.integrations import get_tracer

tracer = get_tracer("langfuse")  # or "langsmith", "datadog", "prometheus"
with tracer.span("evaluate", {"model": "gpt-4o"}):
    ...
```

## LangSmith

Set `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) and optionally
`LANGSMITH_PROJECT`, then:

```python
from checkllm.integrations.langsmith import LangSmithTracer

tracer = LangSmithTracer(project_name="my-evals")
with tracer.span("evaluate", {"model": "gpt-4o"}):
    ...
```

Check results recorded through `tracer.record_check(...)` are posted as
LangSmith feedback entries against the active run.

## LangFuse

Configure `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally
`LANGFUSE_HOST` for self-hosted deployments:

```python
from checkllm.integrations.langfuse import LangFuseTracer

tracer = LangFuseTracer()
with tracer.span("evaluate"):
    ...
tracer.flush()
```

Scores from `record_check` land on the current LangFuse span.

## Datadog

`DatadogTracer` supports three transports:

- `otlp_http` (default) — forwards to the Datadog Agent at
  `http://localhost:4318/v1/traces`
- `otlp_grpc` — forwards to the Datadog Agent at `localhost:4317`
- `ddtrace` — routes through the Datadog Python APM SDK

```python
from checkllm.integrations.datadog import DatadogTracer

tracer = DatadogTracer(service="checkllm-evals", env="prod")
with tracer.span("evaluate"):
    ...
```

Standard Datadog environment variables (`DD_SERVICE`, `DD_ENV`,
`DD_VERSION`, `DD_AGENT_URL`) are respected as defaults.

## Prometheus / Grafana

`PrometheusExporter` exposes counters and histograms that Grafana can
scrape directly:

| Metric                                 | Type      | Labels                  |
|----------------------------------------|-----------|-------------------------|
| `checkllm_evaluations_total`           | Counter   | `metric`, `status`      |
| `checkllm_evaluation_duration_seconds` | Histogram | `metric`                |
| `checkllm_judge_cost_usd_total`        | Counter   | `metric`, `judge`       |
| `checkllm_judge_tokens_total`          | Counter   | `metric`, `judge`, `kind` |

```python
from checkllm.integrations.prometheus import PrometheusExporter

exporter = PrometheusExporter(port=9464)
exporter.start_http_server()

for result in run_evaluation():
    exporter.record_check(result, judge="gpt-4o")
```

When embedding inside an existing HTTP app, pass an external registry
and call `exporter.generate_latest()` from your `/metrics` handler
instead of calling `start_http_server`.

## Dashboard cloud sync

Local SQLite history can be shipped to a central collector (custom
backend, Grafana Agent, webhook, etc.):

```python
from checkllm.history import RunHistory
from checkllm.integrations.cloud_sync import push_to_remote

history = RunHistory()
result = push_to_remote(
    history,
    url="https://collector.example.com/runs",
    token="secret",
    limit=100,
)
print(result.runs_pushed, result.ok)
```

Defaults are driven by `CHECKLLM_REMOTE_URL` and `CHECKLLM_REMOTE_TOKEN`
so CI jobs can opt in without code changes.
