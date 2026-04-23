# Tutorials

A set of hands-on Jupyter notebooks that walk through the main checkllm
workflows. Each notebook is fully runnable end-to-end **without a real API
key** — judged metrics are powered by an in-process `FakeJudge`. A clearly
marked cell in every notebook shows how to swap in a real provider
(`OpenAIJudge`, `AnthropicJudge`, etc.) once you have credentials.

All notebooks live in [`docs/notebooks/`](https://github.com/javierdejesusda/checkllm/tree/main/docs/notebooks)
with their outputs cleared so reviews stay clean.

## Notebooks

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 01 | [Quickstart](https://github.com/javierdejesusda/checkllm/blob/main/docs/notebooks/01_quickstart.ipynb) | Install checkllm, run your first deterministic check, call a judge offline, interpret results. |
| 02 | [RAG evaluation](https://github.com/javierdejesusda/checkllm/blob/main/docs/notebooks/02_rag_evaluation.ipynb) | Faithfulness, context precision/recall, NDCG, and MRR on a toy retrieval pipeline. |
| 03 | [Conversational eval](https://github.com/javierdejesusda/checkllm/blob/main/docs/notebooks/03_conversational_eval.ipynb) | Multi-turn chat evaluation with `ConversationalTestCase` and per-turn scoring. |
| 04 | [Agent trajectory](https://github.com/javierdejesusda/checkllm/blob/main/docs/notebooks/04_agent_trajectory.ipynb) | Tool-use validation, step efficiency, loop detection, and final-answer judging. |
| 05 | [Red teaming](https://github.com/javierdejesusda/checkllm/blob/main/docs/notebooks/05_redteam.ipynb) | Attack generation, OWASP scorecard, vulnerability rollups. |

## Running locally

```bash
pip install checkllm jupyterlab
jupyter lab docs/notebooks/
```

Every notebook starts from a stubbed judge so the first run requires no
secrets. Once you're comfortable, replace `FakeJudge` with a real backend
from `checkllm.judge` and re-run.
