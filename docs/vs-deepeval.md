# CheckLLM vs DeepEval

## TL;DR

DeepEval (\citep{confidentai2024deepeval}) is a general-purpose LLM evaluation
framework with strong judge-based metrics, RAG metrics, and pytest integration.
CheckLLM is focused on **agent-trajectory** evaluation: deterministic, low-latency
scoring of multi-step tool-using agents. The overlap is the agent-trajectory slice;
on RAG and judge metrics DeepEval has the broader feature set.

## When to use DeepEval

- pytest integration is the cleanest in the OSS space; `assert_test` and
  `@pytest.mark.llm_eval` are first-class.
- G-Eval, DAG metrics, and Prometheus-style judge metrics are available
  out of the box, with chain-of-thought rationales surfaced per case.
- RAG-specific metrics: `ContextualPrecisionMetric`,
  `ContextualRecallMetric`, `FaithfulnessMetric`, `AnswerRelevancyMetric`.
- Confident AI cloud platform if you want hosted dashboards, regression
  tracking, and dataset management without standing up your own stack.
- Synthetic dataset generation (`Synthesizer`) is more mature for the
  RAG-Q/A use case than what we ship.

## When to use CheckLLM

- Deterministic 4-axis trajectory scoring (selection, parameters, order,
  efficiency) at **0.034 ms/trajectory** vs **52.24 ms** for DeepEval's
  closest equivalent (`ToolCorrectnessMetric`) — a 1545x latency
  difference.
- **AUROC = 0.938 [0.909, 0.965]** vs synthetic ground truth on 150
  trajectories (DeepEval `ToolCorrectnessMetric`: 0.850 [0.810, 0.895]),
  Holm-corrected p = 0.003.
- First OSS framework with native MCP (Model Context Protocol) metric
  coverage.
- OTel-GenAI ingestion: traces from LangChain, LlamaIndex, CrewAI, and
  PydanticAI all map into one evaluable trajectory format.
- Composite metric weights are Pareto-optimal vs a 1875-cell ablation grid.

## Head-to-head numbers

Source: `benchmarks/paper/results/04_vs_deepeval/summary.json`,
n = 150 trajectories, 1000 bootstrap resamples, deterministic mode (no judge LLM).

| Metric | CheckLLM `TrajectoryMetric` | DeepEval `ToolCorrectnessMetric` |
|---|---|---|
| AUROC | **0.938** [0.909, 0.965] | 0.850 [0.810, 0.895] |
| Spearman ρ | **0.619** [0.528, 0.700] | 0.517 [0.432, 0.604] |
| Mean abs error | 0.447 | **0.390** |
| Latency (ms/traj.) | **0.034** | 52.24 |
| Coverage | 1.00 | 1.00 |
| Cost (USD/traj.) | 0.00 | 0.00 |

**Honesty note: DeepEval has lower mean absolute error.** Its
`ToolCorrectnessMetric` uses LCS-style scoring that polarises outputs
toward {0, 1}, so when it is correct on a binary case the error is near
zero; the trade-off is that it discriminates poorly between partially
correct trajectories, which is why its AUROC and Spearman ρ are lower.
The mean-abs-error difference is **not significant** (Holm-corrected
p ≈ 1.0); the AUROC, Spearman, and latency advantages all are
(p ≤ 0.003).

## If you're already using DeepEval

You can run both in the same pytest suite — they don't fight.

```python
from deepeval.metrics import FaithfulnessMetric
from checkllm.agents import TrajectoryMetric

def test_agent(case):
    assert FaithfulnessMetric().measure(case) >= 0.8   # RAG
    assert TrajectoryMetric().score(case.trace) >= 0.8 # agent
```

Use DeepEval for RAG faithfulness and judge-based eval; use CheckLLM for
the trajectory layer where deterministic, sub-millisecond scoring matters.
