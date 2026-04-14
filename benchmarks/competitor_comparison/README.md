# CheckLLM Competitor Benchmark

Compares CheckLLM against DeepEval, Ragas, and promptfoo on public datasets
with ground truth labels.

## Install

    cd benchmarks/competitor_comparison
    pip install -e ".[dev]"
    npm install -g promptfoo  # for promptfoo adapter

## Run

    checkllm-bench run --framework all --dataset halubench --limit 200 --judge gpt-4o-mini
    checkllm-bench report --out docs/benchmarks/competitor-comparison.md

## Datasets

- HaluBench (PatronusAI/HaluBench) — CC-BY-NC-2.0
- RAGTruth (wandb/RAGTruth-processed) — MIT
- TruthfulQA (truthfulqa/truthful_qa) — Apache 2.0
- JailbreakBench (JailbreakBench/JBB-Behaviors) — MIT
