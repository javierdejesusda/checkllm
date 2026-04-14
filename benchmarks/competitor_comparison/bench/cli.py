from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from bench.adapters.checkllm_adapter import CheckllmAdapter
from bench.adapters.deepeval_adapter import DeepEvalAdapter
from bench.adapters.promptfoo_adapter import PromptfooAdapter
from bench.adapters.ragas_adapter import RagasAdapter
from bench.datasets import (
    load_halubench_from_rows,
    load_hf,
    load_jailbreakbench_from_rows,
    load_ragtruth_from_rows,
    load_truthfulqa_from_rows,
)
from bench.report import build_leaderboard, write_csv, write_html, write_markdown
from bench.runner import BenchmarkRunner, RunSpec
from bench.schema import BenchmarkScore, MetricFamily

app = typer.Typer(name="checkllm-bench")

_DATASET_LOADERS = {
    "halubench": ("PatronusAI/HaluBench", "test", load_halubench_from_rows),
    "ragtruth": ("wandb/RAGTruth-processed", "test", load_ragtruth_from_rows),
    "truthfulqa": ("truthfulqa/truthful_qa", "validation", load_truthfulqa_from_rows),
    "jailbreakbench": ("JailbreakBench/JBB-Behaviors", "harmful", load_jailbreakbench_from_rows),
}


def _build_adapters(frameworks: list[str], judge_model: str):
    from checkllm.providers import create_judge

    out = []
    if "checkllm" in frameworks:
        judge = create_judge("openai", model=judge_model)
        out.append(CheckllmAdapter(judge=judge))
    if "deepeval" in frameworks:
        from deepeval.models import GPTModel
        out.append(DeepEvalAdapter(model=GPTModel(model=judge_model)))
    if "ragas" in frameworks:
        from langchain_openai import ChatOpenAI
        out.append(RagasAdapter(llm=ChatOpenAI(model=judge_model)))
    if "promptfoo" in frameworks:
        out.append(PromptfooAdapter(judge_model=judge_model))
    return out


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset"),
    framework: str = typer.Option("all", "--framework", help="comma-separated or 'all'"),
    family: str = typer.Option("hallucination", "--family"),
    limit: int = typer.Option(200, "--limit"),
    judge: str = typer.Option("gpt-4o-mini", "--judge"),
    out: Path = typer.Option(Path("benchmarks/competitor_comparison/results"), "--out"),
    budget_usd: float = typer.Option(25.0, "--budget-usd"),
) -> None:
    """Run a benchmark sweep and write raw BenchmarkScore JSON to disk."""
    if dataset not in _DATASET_LOADERS:
        raise typer.BadParameter(f"unknown dataset: {dataset}")

    hf_name, split, mapper = _DATASET_LOADERS[dataset]
    rows = load_hf(hf_name, split=split, limit=limit)
    samples = mapper(rows)

    frameworks = (
        ["checkllm", "deepeval", "ragas", "promptfoo"]
        if framework == "all"
        else [f.strip() for f in framework.split(",")]
    )
    adapters = _build_adapters(frameworks, judge)

    families = [MetricFamily(f.strip()) for f in family.split(",")]

    runner = BenchmarkRunner(max_concurrency=8, budget_usd=budget_usd)
    spec = RunSpec(adapters=adapters, samples=samples, families=families, judge_model=judge)
    scores = asyncio.run(runner.run(spec))

    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{dataset}-{family}-{judge}.json"
    raw_path.write_text(
        json.dumps([s.model_dump(mode="json") for s in scores], indent=2), encoding="utf-8"
    )
    typer.echo(f"wrote {len(scores)} scores to {raw_path}")


@app.command()
def report(
    results_dir: Path = typer.Option(Path("benchmarks/competitor_comparison/results"), "--results"),
    out_md: Path = typer.Option(Path("docs/benchmarks/competitor-comparison.md"), "--out-md"),
    out_csv: Path = typer.Option(Path("docs/benchmarks/competitor-comparison.csv"), "--out-csv"),
    out_html: Path = typer.Option(Path("docs/benchmarks/competitor-comparison.html"), "--out-html"),
    labels_file: Path = typer.Option(..., "--labels", help="JSON file: {dataset: {family: {sample_id: label}}}"),
) -> None:
    """Build a leaderboard from raw score JSON files and write MD/CSV/HTML."""
    scores: list[BenchmarkScore] = []
    for path in results_dir.glob("*.json"):
        for item in json.loads(path.read_text(encoding="utf-8")):
            scores.append(BenchmarkScore.model_validate(item))

    raw_labels = json.loads(labels_file.read_text(encoding="utf-8"))
    labels = {
        (dataset, MetricFamily(fam)): fam_labels
        for dataset, fams in raw_labels.items()
        for fam, fam_labels in fams.items()
    }
    rows = build_leaderboard(scores, labels)
    write_markdown(rows, out_md)
    write_csv(rows, out_csv)
    write_html(rows, out_html)
    typer.echo(f"wrote leaderboard to {out_md}, {out_csv}, {out_html}")


@app.command()
def showcase(
    out: Path = typer.Option(Path("docs/benchmarks/checkllm-showcase.md"), "--out"),
    judge: str = typer.Option("gpt-4o-mini", "--judge"),
) -> None:
    """Run CheckLLM-only showcase (compliance, redteam, KG, trajectory) and publish report."""
    from bench.showcase import (
        run_compliance_showcase,
        run_kg_synthesis_showcase,
        run_redteam_showcase,
        run_trajectory_showcase,
        write_showcase_markdown,
    )
    from checkllm.providers import create_judge

    judge_backend = create_judge("openai", model=judge)

    async def dummy_target(prompt: str) -> str:
        return "I cannot help with that request."

    async def gather_reports():
        return [
            await run_compliance_showcase(
                target=dummy_target,
                judge=judge_backend,
                frameworks=["OWASP_LLM_TOP10", "EU_AI_ACT", "HIPAA"],
                attacks_per_type=2,
            ),
            await run_redteam_showcase(
                target=dummy_target,
                judge=judge_backend,
                vulnerability_types=["PROMPT_INJECTION", "JAILBREAK", "PII_LEAKAGE"],
                strategies=["BASE64", "CRESCENDO", "PERSONA"],
                attacks_per_type=2,
            ),
            await run_kg_synthesis_showcase(
                judge=judge_backend,
                documents=[
                    "Python is a programming language created by Guido van Rossum in 1991.",
                    "Rust is a systems programming language focused on safety and concurrency.",
                ],
                num_samples=5,
            ),
            await run_trajectory_showcase(
                judge=judge_backend,
                trajectories=[
                    (
                        [
                            {"tool": "search", "args": {"query": "weather NYC"}},
                            {"tool": "reply", "args": {"text": "Sunny, 72F."}},
                        ],
                        ["search", "reply"],
                    ),
                ],
            ),
        ]

    reports = asyncio.run(gather_reports())
    write_showcase_markdown(reports, out)
    typer.echo(f"wrote showcase to {out}")


if __name__ == "__main__":
    app()
