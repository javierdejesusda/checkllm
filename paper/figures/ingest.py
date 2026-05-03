"""Generate ``generated_tables.tex`` from benchmark summary JSONs.

This script is the sole bridge between the empirical results in
``benchmarks/paper/results/`` and the LaTeX paper in ``paper/checkllm.tex``.
Every numerical claim in the paper is emitted here as a ``\newcommand``
macro, so that the paper can be re-rendered with new numbers by simply
re-running this script.

Usage:
    python paper/figures/ingest.py

The script is idempotent: re-running it on identical inputs produces
byte-identical output.

Raises:
    KeyError: If any expected key is missing from a summary file. The error
        message includes the full JSON path that was being accessed so the
        mismatch can be diagnosed without grepping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "paper" / "results"
OUTPUT_PATH = Path(__file__).resolve().parent / "generated_tables.tex"

LINE_END = r" \\"


def _get(obj: Any, *keys: Any) -> Any:
    """Index into a nested dict/list, raising ``KeyError`` with the full path.

    Args:
        obj: Root container (dict or list) to traverse.
        *keys: Sequence of keys / indices to follow.

    Returns:
        The value at the requested path.

    Raises:
        KeyError: If any key along the path is missing. The error message
            is the dotted path that failed.
    """
    cur = obj
    trail: list[str] = []
    for key in keys:
        trail.append(str(key))
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(".".join(trail)) from exc
    return cur


def _round3(x: float) -> str:
    """Format a number to 3 decimal places."""
    return f"{x:.3f}"


def _round_pct(x: float) -> str:
    """Format a fraction as a percentage to 1 decimal place (no ``%``)."""
    return f"{x * 100:.1f}"


def _round_latency(x: float) -> str:
    """Format a latency in milliseconds to 2 decimal places."""
    return f"{x:.2f}"


def _command(name: str, value: str) -> str:
    """Build a ``\\newcommand{\\name}{value}`` line."""
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _load(path: Path) -> dict[str, Any]:
    """Read a JSON file and return the parsed dict."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _scalar_macros(
    s1: dict[str, Any],
    s2: dict[str, Any],
    s3: dict[str, Any],
    s4: dict[str, Any],
) -> list[str]:
    """Emit ``\\newcommand`` lines for every scalar referenced in the paper."""
    out: list[str] = []

    # C1-zero: controlled noise monotonicity.
    out.append(_command(
        "MonotonicityAirline",
        "passes" if _get(s1, "monotonicity_check", "airline") else "fails",
    ))
    out.append(_command(
        "MonotonicityRetail",
        "passes" if _get(s1, "monotonicity_check", "retail") else "fails",
    ))
    out.append(_command(
        "NoiseSeedsCount",
        str(len(_get(s1, "seeds"))),
    ))
    out.append(_command(
        "NoiseDomainsCount",
        str(len(_get(s1, "domains"))),
    ))
    out.append(_command(
        "NoiseLevelsCount",
        str(len(_get(s1, "noise_levels"))),
    ))
    out.append(_command(
        "NoiseCleanAirlineMean",
        _round3(_get(s1, "domain_noise_means", "airline", "clean")),
    ))
    out.append(_command(
        "NoiseCleanRetailMean",
        _round3(_get(s1, "domain_noise_means", "retail", "clean")),
    ))
    out.append(_command(
        "NoiseSevereAirlineMean",
        _round3(_get(s1, "domain_noise_means", "airline", "severe")),
    ))
    out.append(_command(
        "NoiseSevereRetailMean",
        _round3(_get(s1, "domain_noise_means", "retail", "severe")),
    ))

    # C2-zero: metric vs ground truth.
    out.append(_command(
        "NTrajectories",
        str(_get(s2, "n_trajectories")),
    ))
    out.append(_command(
        "NBootstrap",
        str(_get(s2, "n_bootstrap")),
    ))
    out.append(_command(
        "TruthSeedsCount",
        str(len(_get(s2, "seeds"))),
    ))
    out.append(_command(
        "AurocOverall",
        _round3(_get(s2, "overall", "auroc", "value")),
    ))
    out.append(_command(
        "AurocOverallCiLow",
        _round3(_get(s2, "overall", "auroc", "ci_lower")),
    ))
    out.append(_command(
        "AurocOverallCiHigh",
        _round3(_get(s2, "overall", "auroc", "ci_upper")),
    ))
    out.append(_command(
        "PearsonOverall",
        _round3(_get(s2, "overall", "pearson_r", "value")),
    ))
    out.append(_command(
        "PearsonOverallCiLow",
        _round3(_get(s2, "overall", "pearson_r", "ci_lower")),
    ))
    out.append(_command(
        "PearsonOverallCiHigh",
        _round3(_get(s2, "overall", "pearson_r", "ci_upper")),
    ))
    out.append(_command(
        "SpearmanOverall",
        _round3(_get(s2, "overall", "spearman_rho", "value")),
    ))
    out.append(_command(
        "SpearmanOverallCiLow",
        _round3(_get(s2, "overall", "spearman_rho", "ci_lower")),
    ))
    out.append(_command(
        "SpearmanOverallCiHigh",
        _round3(_get(s2, "overall", "spearman_rho", "ci_upper")),
    ))
    out.append(_command(
        "AurocAirline",
        _round3(_get(s2, "per_domain", "airline", "auroc", "value")),
    ))
    out.append(_command(
        "AurocAirlineCiLow",
        _round3(_get(s2, "per_domain", "airline", "auroc", "ci_lower")),
    ))
    out.append(_command(
        "AurocAirlineCiHigh",
        _round3(_get(s2, "per_domain", "airline", "auroc", "ci_upper")),
    ))
    out.append(_command(
        "AurocRetail",
        _round3(_get(s2, "per_domain", "retail", "auroc", "value")),
    ))
    out.append(_command(
        "AurocRetailCiLow",
        _round3(_get(s2, "per_domain", "retail", "auroc", "ci_lower")),
    ))
    out.append(_command(
        "AurocRetailCiHigh",
        _round3(_get(s2, "per_domain", "retail", "auroc", "ci_upper")),
    ))
    out.append(_command(
        "AurocGateLowerBound",
        _round3(_get(s2, "auroc_gate_lower_bound")),
    ))
    out.append(_command(
        "AurocGatePassedAirline",
        "yes" if _get(s2, "auroc_gate_passed", "airline") else "no",
    ))
    out.append(_command(
        "AurocGatePassedRetail",
        "yes" if _get(s2, "auroc_gate_passed", "retail") else "no",
    ))
    for sub in ("ordering", "loops", "coverage", "unexpected"):
        title = sub.capitalize()
        out.append(_command(
            f"Auroc{title}",
            _round3(_get(s2, "per_metric", sub, "auroc", "value")),
        ))
        out.append(_command(
            f"Auroc{title}CiLow",
            _round3(_get(s2, "per_metric", sub, "auroc", "ci_lower")),
        ))
        out.append(_command(
            f"Auroc{title}CiHigh",
            _round3(_get(s2, "per_metric", sub, "auroc", "ci_upper")),
        ))

    # C3-zero: ablation.
    out.append(_command(
        "AblationNCells",
        str(_get(s3, "n_cells")),
    ))
    out.append(_command(
        "AblationNSkipped",
        str(_get(s3, "n_skipped_degenerate")),
    ))
    out.append(_command(
        "DefaultAuroc",
        _round3(_get(s3, "default_config", "auroc")),
    ))
    out.append(_command(
        "DefaultSpearman",
        _round3(_get(s3, "default_config", "spearman_rho")),
    ))
    out.append(_command(
        "BestAuroc",
        _round3(_get(s3, "best_vs_default_gap", "best_auroc")),
    ))
    out.append(_command(
        "BestSpearman",
        _round3(_get(s3, "best_vs_default_gap", "best_spearman_rho")),
    ))
    out.append(_command(
        "DefaultBestAurocGap",
        _round3(_get(s3, "best_vs_default_gap", "auroc")),
    ))
    out.append(_command(
        "DefaultBestSpearmanGap",
        _round3(_get(s3, "best_vs_default_gap", "spearman_rho")),
    ))
    out.append(_command(
        "DefaultBestGap",
        _round_pct(_get(s3, "best_vs_default_gap", "spearman_rho_relative")),
    ))
    out.append(_command(
        "DefaultIsParetoOptimal",
        "yes" if _get(s3, "default_is_pareto_optimal") else "no",
    ))
    out.append(_command(
        "WeightCorrOrdering",
        _round3(_get(s3, "weight_correlations", "ordering_weight", "with_spearman")),
    ))
    out.append(_command(
        "WeightCorrLoop",
        _round3(_get(s3, "weight_correlations", "loop_weight", "with_spearman")),
    ))
    out.append(_command(
        "WeightCorrCoverage",
        _round3(_get(s3, "weight_correlations", "coverage_weight", "with_spearman")),
    ))
    out.append(_command(
        "WeightCorrUnexpected",
        _round3(_get(s3, "weight_correlations", "unexpected_weight", "with_spearman")),
    ))
    out.append(_command(
        "DefaultOrderingWeight",
        _round3(_get(s3, "default_config", "ordering_weight")),
    ))
    out.append(_command(
        "DefaultLoopWeight",
        _round3(_get(s3, "default_config", "loop_weight")),
    ))
    out.append(_command(
        "DefaultCoverageWeight",
        _round3(_get(s3, "default_config", "coverage_weight")),
    ))
    out.append(_command(
        "DefaultUnexpectedWeight",
        _round3(_get(s3, "default_config", "unexpected_weight")),
    ))
    out.append(_command(
        "DefaultLoopThreshold",
        str(_get(s3, "default_config", "loop_threshold")),
    ))

    # C4-zero: head-to-head vs DeepEval.
    out.append(_command(
        "HHNTrajectories",
        str(_get(s4, "n_trajectories")),
    ))
    out.append(_command(
        "HHNBootstrap",
        str(_get(s4, "n_bootstrap")),
    ))
    out.append(_command(
        "DeepEvalVersion",
        str(_get(s4, "deepeval_version")),
    ))
    out.append(_command(
        "CheckLLMAuroc",
        _round3(_get(s4, "checkllm", "auroc", "value")),
    ))
    out.append(_command(
        "CheckLLMAurocCiLow",
        _round3(_get(s4, "checkllm", "auroc", "ci_lower")),
    ))
    out.append(_command(
        "CheckLLMAurocCiHigh",
        _round3(_get(s4, "checkllm", "auroc", "ci_upper")),
    ))
    out.append(_command(
        "CheckLLMSpearman",
        _round3(_get(s4, "checkllm", "spearman_rho", "value")),
    ))
    out.append(_command(
        "CheckLLMSpearmanCiLow",
        _round3(_get(s4, "checkllm", "spearman_rho", "ci_lower")),
    ))
    out.append(_command(
        "CheckLLMSpearmanCiHigh",
        _round3(_get(s4, "checkllm", "spearman_rho", "ci_upper")),
    ))
    out.append(_command(
        "CheckLLMMae",
        _round3(_get(s4, "checkllm", "mean_abs_error")),
    ))
    out.append(_command(
        "CheckLLMLatency",
        _round_latency(_get(s4, "checkllm", "mean_latency_ms_per_trajectory")),
    ))
    out.append(_command(
        "CheckLLMCoverage",
        _round3(_get(s4, "checkllm", "coverage")),
    ))
    out.append(_command(
        "CheckLLMCost",
        _round3(_get(s4, "checkllm", "cost_usd_per_trajectory")),
    ))
    out.append(_command(
        "DeepEvalAuroc",
        _round3(_get(s4, "deepeval", "auroc", "value")),
    ))
    out.append(_command(
        "DeepEvalAurocCiLow",
        _round3(_get(s4, "deepeval", "auroc", "ci_lower")),
    ))
    out.append(_command(
        "DeepEvalAurocCiHigh",
        _round3(_get(s4, "deepeval", "auroc", "ci_upper")),
    ))
    out.append(_command(
        "DeepEvalSpearman",
        _round3(_get(s4, "deepeval", "spearman_rho", "value")),
    ))
    out.append(_command(
        "DeepEvalSpearmanCiLow",
        _round3(_get(s4, "deepeval", "spearman_rho", "ci_lower")),
    ))
    out.append(_command(
        "DeepEvalSpearmanCiHigh",
        _round3(_get(s4, "deepeval", "spearman_rho", "ci_upper")),
    ))
    out.append(_command(
        "DeepEvalMae",
        _round3(_get(s4, "deepeval", "mean_abs_error")),
    ))
    out.append(_command(
        "DeepEvalLatency",
        _round_latency(_get(s4, "deepeval", "mean_latency_ms_per_trajectory")),
    ))
    out.append(_command(
        "DeepEvalCoverage",
        _round3(_get(s4, "deepeval", "coverage")),
    ))
    out.append(_command(
        "DeepEvalCost",
        _round3(_get(s4, "deepeval", "cost_usd_per_trajectory")),
    ))

    latency_ratio = (
        _get(s4, "deepeval", "mean_latency_ms_per_trajectory")
        / _get(s4, "checkllm", "mean_latency_ms_per_trajectory")
    )
    out.append(_command(
        "PaperHeadToHeadLatencyRatio",
        f"{latency_ratio:,.0f}".replace(",", "{,}"),
    ))

    claims = _get(s4, "head_to_head", "claims")
    by_name = {c["name"]: c for c in claims}
    for short, full in [
        ("AurocClaim", "checkllm_auroc_gt_deepeval"),
        ("SpearmanClaim", "checkllm_spearman_gt_deepeval"),
        ("MaeClaim", "checkllm_abs_error_lt_deepeval"),
        ("LatencyClaim", "checkllm_latency_lt_deepeval"),
    ]:
        c = by_name[full]
        out.append(_command(
            f"{short}PvalueRaw",
            f"{c['p_value_raw']:.3g}",
        ))
        out.append(_command(
            f"{short}PvalueHolm",
            f"{c['p_value_holm_corrected']:.3g}",
        ))
        out.append(_command(
            f"{short}Significant",
            "yes" if c["significant_at_0_05"] else "no",
        ))
    out.append(_command(
        "HHClaimsCount",
        str(_get(s4, "head_to_head", "n_claims")),
    ))
    n_significant = sum(1 for c in claims if c["significant_at_0_05"])
    out.append(_command("HHClaimsSignificant", str(n_significant)))
    out.append(_command(
        "HolmAlpha",
        _round3(_get(s4, "head_to_head", "holm_alpha")),
    ))

    return out


def _table_noise_monotonicity(s1: dict[str, Any]) -> str:
    """LaTeX tabular for the noise-vs-overall-score sweep, both domains."""
    levels = _get(s1, "noise_levels")
    rows: list[str] = []
    for level in levels:
        a = _round3(_get(s1, "domain_noise_means", "airline", level))
        r = _round3(_get(s1, "domain_noise_means", "retail", level))
        rows.append(f"{level} & {a} & {r}{LINE_END}")
    body = "\n".join(rows)
    table = (
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        f"Noise level & Airline & Retail{LINE_END}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return _command("TableNoiseMonotonicity", table)


def _table_metric_vs_truth(s2: dict[str, Any]) -> str:
    """LaTeX tabular reporting AUROC, Pearson r, Spearman rho per sub-score."""
    rows: list[str] = []
    for sub in ("overall", "ordering", "unexpected", "coverage", "loops"):
        auroc = _get(s2, "per_metric", sub, "auroc")
        pearson = _get(s2, "per_metric", sub, "pearson_r")
        spearman = _get(s2, "per_metric", sub, "spearman_rho")
        rows.append(
            f"{sub} & "
            f"{_round3(auroc['value'])} "
            f"[{_round3(auroc['ci_lower'])}, {_round3(auroc['ci_upper'])}] & "
            f"{_round3(pearson['value'])} "
            f"[{_round3(pearson['ci_lower'])}, {_round3(pearson['ci_upper'])}] & "
            f"{_round3(spearman['value'])} "
            f"[{_round3(spearman['ci_lower'])}, {_round3(spearman['ci_upper'])}]"
            f"{LINE_END}"
        )
    body = "\n".join(rows)
    table = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        f"Sub-score & AUROC [95\\% CI] & Pearson $r$ [95\\% CI] & "
        f"Spearman $\\rho$ [95\\% CI]{LINE_END}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return _command("TableMetricVsTruth", table)


def _table_ablation_top_five(s3: dict[str, Any]) -> str:
    """LaTeX tabular: top-5 grid cells by Spearman vs default."""
    top = _get(s3, "top_5_by_spearman")
    rows: list[str] = []
    for cell in top:
        rows.append(
            f"{_round3(cell['ordering_weight'])} & "
            f"{_round3(cell['loop_weight'])} & "
            f"{_round3(cell['coverage_weight'])} & "
            f"{_round3(cell['unexpected_weight'])} & "
            f"{cell['loop_threshold']} & "
            f"{_round3(cell['auroc'])} & "
            f"{_round3(cell['spearman_rho'])}{LINE_END}"
        )
    default = _get(s3, "default_config")
    rows.append(
        "\\midrule\n"
        f"{_round3(default['ordering_weight'])} & "
        f"{_round3(default['loop_weight'])} & "
        f"{_round3(default['coverage_weight'])} & "
        f"{_round3(default['unexpected_weight'])} & "
        f"{default['loop_threshold']} & "
        f"{_round3(default['auroc'])} & "
        f"{_round3(default['spearman_rho'])}{LINE_END}"
    )
    body = "\n".join(rows)
    table = (
        "\\begin{tabular}{ccccccc}\n"
        "\\toprule\n"
        f"$w_o$ & $w_l$ & $w_c$ & $w_u$ & $T_{{\\text{{loop}}}}$ "
        f"& AUROC & Spearman $\\rho${LINE_END}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return _command("TableAblationTopFive", table)


def _table_head_to_head(s4: dict[str, Any]) -> str:
    """LaTeX tabular: CheckLLM vs DeepEval headline metrics."""
    cl = _get(s4, "checkllm")
    de = _get(s4, "deepeval")
    rows = [
        "AUROC & "
        f"{_round3(cl['auroc']['value'])} "
        f"[{_round3(cl['auroc']['ci_lower'])}, {_round3(cl['auroc']['ci_upper'])}] & "
        f"{_round3(de['auroc']['value'])} "
        f"[{_round3(de['auroc']['ci_lower'])}, {_round3(de['auroc']['ci_upper'])}]"
        f"{LINE_END}",
        "Spearman $\\rho$ & "
        f"{_round3(cl['spearman_rho']['value'])} "
        f"[{_round3(cl['spearman_rho']['ci_lower'])}, "
        f"{_round3(cl['spearman_rho']['ci_upper'])}] & "
        f"{_round3(de['spearman_rho']['value'])} "
        f"[{_round3(de['spearman_rho']['ci_lower'])}, "
        f"{_round3(de['spearman_rho']['ci_upper'])}]{LINE_END}",
        "Mean abs.\\ error & "
        f"{_round3(cl['mean_abs_error'])} & {_round3(de['mean_abs_error'])}{LINE_END}",
        "Latency (ms / traj.) & "
        f"{_round_latency(cl['mean_latency_ms_per_trajectory'])} & "
        f"{_round_latency(de['mean_latency_ms_per_trajectory'])}{LINE_END}",
        "Coverage & "
        f"{_round3(cl['coverage'])} & {_round3(de['coverage'])}{LINE_END}",
        "Cost (USD / traj.) & "
        f"{_round3(cl['cost_usd_per_trajectory'])} & "
        f"{_round3(de['cost_usd_per_trajectory'])}{LINE_END}",
    ]
    body = "\n".join(rows)
    table = (
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        f"Metric & CheckLLM & DeepEval{LINE_END}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return _command("TableHeadToHead", table)


def main() -> None:
    """Read result summaries and write ``generated_tables.tex``."""
    s1 = _load(RESULTS_DIR / "01_controlled_noise" / "summary.json")
    s2 = _load(RESULTS_DIR / "02_metric_vs_truth" / "summary.json")
    s3 = _load(RESULTS_DIR / "03_ablation" / "summary.json")
    s4 = _load(RESULTS_DIR / "04_vs_deepeval" / "summary.json")

    parts: list[str] = [
        "% This file is generated by paper/figures/ingest.py.",
        "% Do not edit by hand: regenerate with `python paper/figures/ingest.py`.",
        "",
        "% --- scalar macros ---",
    ]
    parts.extend(_scalar_macros(s1, s2, s3, s4))
    parts.append("")
    parts.append("% --- tabular blocks ---")
    parts.append(_table_noise_monotonicity(s1))
    parts.append(_table_metric_vs_truth(s2))
    parts.append(_table_ablation_top_five(s3))
    parts.append(_table_head_to_head(s4))
    parts.append("")

    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
