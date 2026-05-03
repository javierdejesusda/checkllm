"""Task C4-zero: head-to-head vs DeepEval ``ToolCorrectnessMetric``.

This experiment scores the same C1-zero StubAgent trajectories with two
deterministic, zero-API-cost metrics:

1. CheckLLM's :class:`TrajectoryMetric` (overall sub-score).
2. DeepEval's :class:`ToolCorrectnessMetric` in deterministic mode
   (``async_mode=False``, no ``available_tools``, ``should_consider_ordering=True``
   so it factors action ordering analogously to CheckLLM).

For each trajectory we record both metrics' score and per-call latency,
plus the synthetic binary label (``label = 1`` iff
``noise_level == "clean"``). Aggregate statistics:

* Pearson r (point-biserial), Spearman rho, AUROC vs label, with 95%
  percentile bootstrap CIs (n=1000).
* Mean per-trajectory absolute error: ``mean(abs(score - label))``.
* Mean per-trajectory latency in milliseconds (timed around just the
  metric scoring call).
* Coverage = fraction of trajectories the metric returned a non-error
  score for.

Then four head-to-head claims are tested with raw + Holm-Bonferroni
corrected p-values:

1. CheckLLM AUROC > DeepEval AUROC (one-sided bootstrap).
2. CheckLLM Spearman rho > DeepEval Spearman rho (one-sided bootstrap).
3. CheckLLM mean abs-error < DeepEval mean abs-error (paired t-test).
4. CheckLLM mean latency < DeepEval mean latency (paired t-test).

Outputs in ``output_dir``:

* ``per_trajectory.jsonl`` -- one row per trajectory with both metrics'
  score, latency, and absolute error.
* ``summary.json`` -- aggregate statistics + head-to-head claims block.
* ``manifest.json`` -- per-trajectory provenance with a ``framework``
  column ("checkllm" or "deepeval"); doubles trajectory count.

Run via:

    python benchmarks/paper/experiments/04_vs_deepeval.py \
        --input-file benchmarks/paper/results/01_controlled_noise/trajectories.jsonl \
        --output-dir benchmarks/paper/results/04_vs_deepeval

Note: because the file name starts with a digit it is not a valid
Python identifier; ``python -m benchmarks.paper.experiments.04_vs_deepeval``
is therefore not supported. Invoke the script by file path instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from checkllm.metrics.trajectory_metric import TrajectoryMetric  # noqa: E402

import deepeval  # noqa: E402
from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402


EXPERIMENT_ID = "04_vs_deepeval"
DEFAULT_N_BOOTSTRAP = 1000
HOLM_ALPHA = 0.05

CLAIM_NAMES: tuple[str, ...] = (
    "checkllm_auroc_gt_deepeval",
    "checkllm_spearman_gt_deepeval",
    "checkllm_abs_error_lt_deepeval",
    "checkllm_latency_lt_deepeval",
)


@dataclass(frozen=True)
class TrajectoryRow:
    """One trajectory loaded from the C1-zero ``trajectories.jsonl``.

    Attributes:
        domain: ``"airline"`` or ``"retail"``.
        task_id: Tau-bench task identifier (e.g. ``"airline-synth-1"``).
        seed: Stub-agent seed used when generating the trajectory.
        noise_level: One of ``clean / light / medium / heavy / severe``.
        predicted_actions: List of ``{name, arguments}`` dicts produced
            by the agent.
        reference_actions: List of ``{name, arguments}`` dicts from the
            tau-bench gold trajectory.
    """

    domain: str
    task_id: str
    seed: int
    noise_level: str
    predicted_actions: list[dict[str, Any]]
    reference_actions: list[dict[str, Any]]


def _load_trajectories(input_file: Path) -> list[TrajectoryRow]:
    """Load all rows from a C1-zero-style ``trajectories.jsonl``.

    Args:
        input_file: Path to a JSONL file with rows containing
            ``domain``, ``task_id``, ``seed``, ``noise_level``,
            ``predicted_actions``, and ``reference_actions``.

    Returns:
        A list of :class:`TrajectoryRow`.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
    """
    rows: list[TrajectoryRow] = []
    with Path(input_file).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                TrajectoryRow(
                    domain=str(obj["domain"]),
                    task_id=str(obj["task_id"]),
                    seed=int(obj["seed"]),
                    noise_level=str(obj["noise_level"]),
                    predicted_actions=list(obj["predicted_actions"]),
                    reference_actions=list(obj["reference_actions"]),
                )
            )
    return rows


def _to_deepeval_tool_calls(
    actions: list[dict[str, Any]],
) -> list[ToolCall]:
    """Convert CheckLLM-style actions to DeepEval :class:`ToolCall`s.

    Args:
        actions: List of dicts with ``name`` and optional ``arguments``.

    Returns:
        DeepEval :class:`ToolCall` instances. ``input_parameters`` is
        populated from ``arguments`` so DeepEval can use it if
        ``ToolCallParams.INPUT_PARAMETERS`` is requested (we don't, but
        we include it to keep the test cases honest).
    """
    out: list[ToolCall] = []
    for action in actions:
        name = str(action.get("name", ""))
        args = action.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCall(name=name, input_parameters=args))
    return out


def _score_checkllm(row: TrajectoryRow) -> tuple[float, float]:
    """Score one trajectory with CheckLLM's TrajectoryMetric.

    Args:
        row: Trajectory to score.

    Returns:
        Tuple ``(score, latency_ms)`` with ``score`` clamped to
        ``[0.0, 1.0]`` and ``latency_ms`` measured around just the
        ``compute_subscores`` call.
    """
    expected = [str(a.get("name", "")) for a in row.reference_actions]
    actual = [str(a.get("name", "")) for a in row.predicted_actions]
    metric = TrajectoryMetric(expected_trajectory=expected)
    t0 = time.perf_counter()
    sub = metric.compute_subscores(actual)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    score = max(0.0, min(1.0, float(sub.overall)))
    return score, dt_ms


def _score_deepeval(row: TrajectoryRow) -> tuple[float, float]:
    """Score one trajectory with DeepEval's ToolCorrectnessMetric.

    Uses the synchronous, non-LLM deterministic path:
    ``async_mode=False`` and no ``available_tools``. We enable
    ``should_consider_ordering=True`` so DeepEval factors call ordering
    analogously to CheckLLM (otherwise DeepEval is order-blind, which
    would be an unfair head-to-head).

    Args:
        row: Trajectory to score.

    Returns:
        Tuple ``(score, latency_ms)`` with ``score`` clamped to
        ``[0.0, 1.0]`` and ``latency_ms`` measured around just the
        ``measure`` call.
    """
    metric = ToolCorrectnessMetric(
        async_mode=False,
        include_reason=False,
        should_consider_ordering=True,
        verbose_mode=False,
    )
    test_case = LLMTestCase(
        input=row.task_id,
        actual_output="",
        tools_called=_to_deepeval_tool_calls(row.predicted_actions),
        expected_tools=_to_deepeval_tool_calls(row.reference_actions),
    )
    t0 = time.perf_counter()
    score = metric.measure(
        test_case,
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    score = max(0.0, min(1.0, float(score)))
    return score, dt_ms


def _auroc_mann_whitney(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """AUROC via the Mann-Whitney U statistic (no sklearn dependency).

    Args:
        scores: 1-D array of real-valued classifier scores.
        labels: 1-D array of binary labels in ``{0, 1}`` aligned with
            ``scores``.

    Returns:
        AUROC in ``[0, 1]``. Returns ``nan`` if a label class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    sum_ranks_pos = float(ranks[labels == 1].sum())
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _pearson_r(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pearson r between scores and binary labels (point-biserial)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.std() == 0.0 or labels.std() == 0.0:
        return float("nan")
    return float(stats.pearsonr(scores, labels).statistic)


def _spearman_rho(scores: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rho between scores and binary labels."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.std() == 0.0 or labels.std() == 0.0:
        return float("nan")
    result = stats.spearmanr(scores, labels)
    statistic = getattr(result, "statistic", None)
    if statistic is None:
        statistic = result[0]  # type: ignore[index]
    return float(statistic)


def _bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute a 95% percentile bootstrap CI on a paired statistic.

    Args:
        scores: 1-D scores array.
        labels: 1-D binary labels aligned with ``scores``.
        statistic_fn: Callable taking ``(scores, labels)`` and returning
            a scalar statistic.
        n_bootstrap: Number of bootstrap resamples.
        rng: NumPy random Generator for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)``. ``(nan, nan)`` if undefined.
    """

    def _vec(s: np.ndarray, l_: np.ndarray) -> float:
        return statistic_fn(s, l_)

    try:
        result = stats.bootstrap(
            (scores, labels),
            _vec,
            n_resamples=n_bootstrap,
            paired=True,
            method="percentile",
            confidence_level=0.95,
            vectorized=False,
            random_state=rng,
        )
    except ValueError:
        return float("nan"), float("nan")

    ci_lower = float(result.confidence_interval.low)
    ci_upper = float(result.confidence_interval.high)
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        dist = np.asarray(result.bootstrap_distribution, dtype=float)
        dist = dist[~np.isnan(dist)]
        if dist.size == 0:
            return float("nan"), float("nan")
        ci_lower = float(np.percentile(dist, 2.5))
        ci_upper = float(np.percentile(dist, 97.5))
    return ci_lower, ci_upper


def _statistics_block(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Build the {pearson_r, spearman_rho, auroc} block with 95% CIs."""
    block: dict[str, dict[str, float]] = {}
    for name, fn in (
        ("pearson_r", _pearson_r),
        ("spearman_rho", _spearman_rho),
        ("auroc", _auroc_mann_whitney),
    ):
        value = fn(scores, labels)
        ci_lower, ci_upper = _bootstrap_ci(
            scores, labels, fn, n_bootstrap=n_bootstrap, rng=rng
        )
        block[name] = {
            "value": float(value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
    return block


def _bootstrap_diff_p_value(
    a_scores: np.ndarray,
    b_scores: np.ndarray,
    labels: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """One-sided bootstrap p-value for ``stat(a) - stat(b) > 0``.

    Resamples paired indices into ``(a_scores, b_scores, labels)`` and
    counts the fraction of resamples where the observed difference is
    less than or equal to zero.

    Args:
        a_scores: First metric's scores (the "claimed-better" side).
        b_scores: Second metric's scores.
        labels: Binary labels aligned with both score arrays.
        statistic_fn: Callable ``(scores, labels) -> float``.
        n_bootstrap: Number of resamples.
        rng: NumPy random Generator.

    Returns:
        One-sided p-value in ``[0, 1]``. Returns ``1.0`` if all
        resamples produced undefined statistics (no evidence).
    """
    n = a_scores.size
    diffs = np.empty(n_bootstrap, dtype=float)
    diffs.fill(np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(low=0, high=n, size=n)
        a_stat = statistic_fn(a_scores[idx], labels[idx])
        b_stat = statistic_fn(b_scores[idx], labels[idx])
        diffs[i] = a_stat - b_stat
    valid = diffs[~np.isnan(diffs)]
    if valid.size == 0:
        return 1.0
    n_fail = int((valid <= 0).sum())
    return float((n_fail + 1) / (valid.size + 1))


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of raw p-values.

    Sorts p-values ascending; multiplies each by ``(n - i)`` where
    ``i`` is the 0-indexed rank; takes a running max so corrected values
    are monotonic; clips at 1.0; returns in original order.

    Args:
        p_values: Raw p-values in original claim order.

    Returns:
        List of Holm-corrected p-values in original claim order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected_sorted: list[float] = []
    running_max = 0.0
    for i, (_, p) in enumerate(indexed):
        adjusted = p * (n - i)
        running_max = max(running_max, adjusted)
        corrected_sorted.append(min(running_max, 1.0))
    out = [0.0] * n
    for rank, (orig_index, _) in enumerate(indexed):
        out[orig_index] = corrected_sorted[rank]
    return out


def _one_sided_paired_p(higher: np.ndarray, lower: np.ndarray) -> float:
    """One-sided paired t-test p-value for ``mean(higher) > mean(lower)``.

    Args:
        higher: Values claimed to be larger on average.
        lower: Values claimed to be smaller on average.

    Returns:
        One-sided p-value. ``1.0`` if every paired difference is zero
        (degenerate; t-test would NaN).
    """
    diffs = np.asarray(higher, dtype=float) - np.asarray(lower, dtype=float)
    if np.allclose(diffs, 0.0):
        return 1.0
    if diffs.std() == 0.0:
        return 0.0 if diffs.mean() > 0 else 1.0
    result = stats.ttest_rel(higher, lower)
    p_two_sided = float(result.pvalue)
    statistic = float(result.statistic)
    if statistic > 0:
        return p_two_sided / 2.0
    return 1.0 - p_two_sided / 2.0


def _resolve_deepeval_version() -> str:
    """Read DeepEval's installed version string.

    Tries ``deepeval.__version__`` first, then ``importlib.metadata``.
    Falls back to ``"unknown"`` if neither path resolves.

    Returns:
        Version string (typically semver-ish like ``"3.9.7"``).
    """
    version = getattr(deepeval, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib_metadata.version("deepeval")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _build_summary(
    per_trajectory_rows: list,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """Aggregate per-trajectory rows into the head-to-head summary.

    Args:
        per_trajectory_rows: One dict per trajectory carrying both
            metrics' scores, latencies, and absolute errors.
        n_bootstrap: Number of bootstrap resamples for CIs and
            difference p-values.
        rng: NumPy random Generator.

    Returns:
        Summary dict matching the spec's ``summary.json`` shape.
    """
    labels = np.asarray([r["label"] for r in per_trajectory_rows], dtype=int)
    ck_scores = np.asarray(
        [r["checkllm_score"] for r in per_trajectory_rows], dtype=float
    )
    de_scores = np.asarray(
        [r["deepeval_score"] for r in per_trajectory_rows], dtype=float
    )
    ck_lat = np.asarray(
        [r["checkllm_latency_ms"] for r in per_trajectory_rows], dtype=float
    )
    de_lat = np.asarray(
        [r["deepeval_latency_ms"] for r in per_trajectory_rows], dtype=float
    )
    ck_abs = np.asarray(
        [r["checkllm_abs_error"] for r in per_trajectory_rows], dtype=float
    )
    de_abs = np.asarray(
        [r["deepeval_abs_error"] for r in per_trajectory_rows], dtype=float
    )

    ck_block = _statistics_block(ck_scores, labels, n_bootstrap, rng)
    de_block = _statistics_block(de_scores, labels, n_bootstrap, rng)

    ck_block.update(
        {
            "coverage": float(
                np.mean([r["checkllm_score"] is not None for r in per_trajectory_rows])
            ),
            "mean_latency_ms_per_trajectory": float(np.mean(ck_lat)),
            "mean_abs_error": float(np.mean(ck_abs)),
            "cost_usd_per_trajectory": 0.0,
        }
    )
    de_block.update(
        {
            "coverage": float(
                np.mean([r["deepeval_score"] is not None for r in per_trajectory_rows])
            ),
            "mean_latency_ms_per_trajectory": float(np.mean(de_lat)),
            "mean_abs_error": float(np.mean(de_abs)),
            "cost_usd_per_trajectory": 0.0,
        }
    )

    p_auroc = _bootstrap_diff_p_value(
        ck_scores, de_scores, labels, _auroc_mann_whitney,
        n_bootstrap=n_bootstrap, rng=rng,
    )
    p_spearman = _bootstrap_diff_p_value(
        ck_scores, de_scores, labels, _spearman_rho,
        n_bootstrap=n_bootstrap, rng=rng,
    )
    p_abs = _one_sided_paired_p(de_abs, ck_abs)
    p_lat = _one_sided_paired_p(de_lat, ck_lat)

    raw_p_values = [p_auroc, p_spearman, p_abs, p_lat]
    corrected = _holm_bonferroni(raw_p_values)

    claims = [
        {
            "name": CLAIM_NAMES[i],
            "p_value_raw": float(raw_p_values[i]),
            "p_value_holm_corrected": float(corrected[i]),
            "significant_at_0_05": bool(corrected[i] < HOLM_ALPHA),
        }
        for i in range(len(CLAIM_NAMES))
    ]

    return {
        "experiment_id": EXPERIMENT_ID,
        "n_trajectories": len(per_trajectory_rows),
        "deepeval_version": _resolve_deepeval_version(),
        "checkllm": ck_block,
        "deepeval": de_block,
        "head_to_head": {
            "claims": claims,
            "holm_alpha": HOLM_ALPHA,
            "n_claims": len(CLAIM_NAMES),
        },
        "n_bootstrap": n_bootstrap,
    }


def run_experiment(
    input_file,
    output_dir,
    limit_trajectories=None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict:
    """Run the full head-to-head experiment.

    Args:
        input_file: Path to a C1-zero ``trajectories.jsonl`` file.
        output_dir: Output directory for ``per_trajectory.jsonl``,
            ``summary.json``, and ``manifest.json``. Created if missing;
            existing files are overwritten.
        limit_trajectories: Optional cap on number of trajectories
            (truncates from the start of the file). For smoke testing.
        n_bootstrap: Resamples for both CIs and bootstrap diff
            p-values.

    Returns:
        A small CLI-friendly summary dict.
    """
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    per_traj_path = output_root / "per_trajectory.jsonl"
    summary_path = output_root / "summary.json"
    manifest_path = output_root / "manifest.json"

    rows = _load_trajectories(Path(input_file))
    if limit_trajectories is not None:
        rows = rows[:limit_trajectories]

    rng = np.random.default_rng(seed=20260503)
    deepeval_version = _resolve_deepeval_version()

    per_trajectory_rows = []
    manifest_rows = []
    for row in rows:
        label = 1 if row.noise_level == "clean" else 0
        ck_score, ck_lat = _score_checkllm(row)
        de_score, de_lat = _score_deepeval(row)
        ck_abs = abs(ck_score - label)
        de_abs = abs(de_score - label)
        timestamp = datetime.now(timezone.utc).isoformat()

        per_trajectory_rows.append(
            {
                "domain": row.domain,
                "task_id": row.task_id,
                "seed": row.seed,
                "noise_level": row.noise_level,
                "label": label,
                "checkllm_score": float(ck_score),
                "checkllm_latency_ms": float(ck_lat),
                "checkllm_abs_error": float(ck_abs),
                "deepeval_score": float(de_score),
                "deepeval_latency_ms": float(de_lat),
                "deepeval_abs_error": float(de_abs),
            }
        )

        for framework, score, lat, abs_err in (
            ("checkllm", ck_score, ck_lat, ck_abs),
            ("deepeval", de_score, de_lat, de_abs),
        ):
            manifest_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": "stub",
                    "benchmark": "tau_bench",
                    "framework": framework,
                    "domain": row.domain,
                    "seed": row.seed,
                    "task_id": row.task_id,
                    "noise_level": row.noise_level,
                    "trajectory_score": float(score),
                    "label": label,
                    "abs_error": float(abs_err),
                    "reference_action_count": len(row.reference_actions),
                    "predicted_action_count": len(row.predicted_actions),
                    "timestamp_utc": timestamp,
                    "model_version_sha": None,
                    "benchmark_sha": None,
                    "temperature": None,
                    "mean_latency_ms": float(lat),
                    "total_cost_usd": 0.0,
                }
            )

    with per_traj_path.open("w", encoding="utf-8") as fh:
        for r in per_trajectory_rows:
            fh.write(json.dumps(r) + "\n")

    summary = _build_summary(
        per_trajectory_rows, n_bootstrap=n_bootstrap, rng=rng
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema": "https://checkllm.dev/schemas/paper_manifest/v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deepeval_version": deepeval_version,
        "input_file": str(Path(input_file)),
        "limit_trajectories": limit_trajectories,
        "n_bootstrap": n_bootstrap,
        "rows": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {
        "n_trajectories": len(per_trajectory_rows),
        "output_dir": str(output_root),
        "deepeval_version": deepeval_version,
        "checkllm_auroc": summary["checkllm"]["auroc"]["value"],
        "deepeval_auroc": summary["deepeval"]["auroc"]["value"],
        "claims_significant_after_correction": [
            c["name"] for c in summary["head_to_head"]["claims"]
            if c["significant_at_0_05"]
        ],
    }


def _parse_args(argv=None):
    """Build the argparse namespace for the CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Head-to-head vs DeepEval ToolCorrectnessMetric (Task C4-zero)."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help=(
            "Path to a C1-zero trajectories.jsonl file (must contain "
            "predicted_actions and reference_actions per row)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory to write per_trajectory.jsonl, summary.json, "
            "manifest.json."
        ),
    )
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=None,
        help=(
            "Truncate to the first N trajectories from the input file "
            "(smoke testing)."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=(
            "Number of bootstrap resamples for 95%% CIs and head-to-head "
            "difference p-values (default: 1000)."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Console entry point.

    Args:
        argv: Optional explicit argv for testing. ``None`` uses ``sys.argv``.

    Returns:
        Exit code; ``0`` on success.
    """
    args = _parse_args(argv)
    summary = run_experiment(
        input_file=args.input_file,
        output_dir=args.output_dir,
        limit_trajectories=args.limit_trajectories,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())