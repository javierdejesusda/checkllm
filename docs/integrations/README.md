# Framework integration adapters

CheckLLM ships lightweight adapters that translate native agent-run
objects from popular agent frameworks into a `checkllm.agents.AgentTestCase`,
so they can be scored by `TrajectoryMetric`.

Each adapter:

- Is a single function (`to_checkllm_test_case`) plus a tool-call helper
  (`to_checkllm_tool_calls`).
- Imports the framework lazily — installing CheckLLM does **not** add
  any of these frameworks as a runtime dependency.
- Is deterministic: no LLM calls, no I/O, no network.

## Adapters

| Framework | Module | Example |
| --- | --- | --- |
| LangChain (`AgentExecutor`) | `checkllm.integrations.langchain` | [langchain_example.py](langchain_example.py) |
| LlamaIndex (`AgentChatResponse`) | `checkllm.integrations.llama_index` | [llama_index_example.py](llama_index_example.py) |
| CrewAI (`CrewOutput` / `TaskOutput`) | `checkllm.integrations.crewai` | [crewai_example.py](crewai_example.py) |
| pydantic-ai (`RunResult`) | `checkllm.integrations.pydantic_ai` | [pydantic_ai_example.py](pydantic_ai_example.py) |

## Pre-drafted upstream PRs

The same adapters are also packaged for inclusion in each upstream
repo, with a short PR description explaining the integration. Drafts
live under [`pr_drafts/`](pr_drafts/) and are intentionally **not yet
opened** — review them before any of these PRs goes live.

- [`pr_drafts/langchain.md`](pr_drafts/langchain.md)
- [`pr_drafts/llama_index.md`](pr_drafts/llama_index.md)
- [`pr_drafts/crewai.md`](pr_drafts/crewai.md)
- [`pr_drafts/pydantic_ai.md`](pr_drafts/pydantic_ai.md)

## Quick example

```python
from langchain.agents import AgentExecutor  # your real executor
from checkllm.integrations.langchain import to_checkllm_test_case
from checkllm.metrics.trajectory_metric import TrajectoryMetric

run = executor.invoke({"input": "How tall is Everest?"})
case = to_checkllm_test_case(run, query="How tall is Everest?")
metric = TrajectoryMetric(expected_trajectory=["search", "calculator"])
print(metric.evaluate(case.tool_calls).reasoning)
```
