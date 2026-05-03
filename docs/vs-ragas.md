# CheckLLM vs Ragas

## TL;DR

Ragas (\citep{es2024ragas}) is the de-facto evaluation library for
**Retrieval-Augmented Generation**. CheckLLM is focused on
**agent-trajectory** evaluation. The two are largely orthogonal: if your
system is a RAG pipeline, reach for Ragas; if it is a tool-using agent,
reach for CheckLLM. The slice that overlaps is "agents that also retrieve."

## When to use Ragas

- The reference RAG metric vocabulary: faithfulness, answer relevancy,
  context precision, context recall — many practitioners adopt Ragas
  names as the lingua franca for RAG eval.
- Well-cited methodology, with the original paper covering metric design
  and validation against human judgements (\citep{es2024ragas}).
- Tight integration with LangChain RAG chains and LlamaIndex; the
  `evaluate()` entry point accepts native chain outputs.
- Synthetic test-set generation tailored for RAG (question types,
  evolutionary complexity).
- Active community of RAG-focused contributors; new RAG metrics tend to
  land here first.

## When to use CheckLLM

- Deterministic 4-axis trajectory scoring (selection, parameters, order,
  efficiency) at **0.034 ms/trajectory** with no judge-LLM cost.
- **AUROC = 0.938 [0.909, 0.965]** vs synthetic ground truth on 150
  trajectories — measured against the closest agent-eval competitor
  (DeepEval `ToolCorrectnessMetric`).
- First OSS framework with native MCP (Model Context Protocol) metric
  coverage.
- OTel-GenAI ingestion: traces from LangChain, LlamaIndex, CrewAI, and
  PydanticAI all map into one evaluable trajectory format.
- Composite metric weights are Pareto-optimal vs a 1875-cell ablation grid.

## Where they don't overlap

Ragas answers **"is this RAG answer faithful and relevant given the
retrieved context?"** It assumes a single (question, contexts, answer)
tuple per case.

CheckLLM answers **"did this agent select the right tools, call them
with the right parameters, in a sensible order, without redundant
loops?"** It assumes a multi-step trace with tool calls, observations,
and (optionally) a final answer.

For an agentic-RAG system you typically want **both**: Ragas to grade
the final grounded answer, CheckLLM to grade the retrieval/tool-use
trajectory that produced it.

## If you're already using Ragas

Run both. Ragas grades the answer; CheckLLM grades the path.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from checkllm.agents import TrajectoryMetric

ragas_score = evaluate(rag_dataset, metrics=[faithfulness, answer_relevancy])
traj_score  = TrajectoryMetric().score(agent_trace)
```

The two scores answer different questions. Reporting both is the
honest move for any retrieval-using agent.
