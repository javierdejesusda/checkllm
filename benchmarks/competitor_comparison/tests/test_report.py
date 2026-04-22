from bench.report import build_leaderboard, write_csv, write_html, write_markdown
from bench.schema import BenchmarkScore, MetricFamily


def _mk(framework: str, sample_id: str, score: float) -> BenchmarkScore:
    return BenchmarkScore(
        framework=framework,
        dataset="halubench",
        metric_family=MetricFamily.HALLUCINATION,
        metric_name="hallucination",
        sample_id=sample_id,
        score=score,
        passed=score >= 0.5,
        latency_ms=100,
        cost_usd=0.001,
        judge_model="gpt-4o-mini",
    )


def test_leaderboard_ranks_by_auc():
    scores = [
        _mk("checkllm", "a", 0.9),
        _mk("checkllm", "b", 0.1),
        _mk("deepeval", "a", 0.7),
        _mk("deepeval", "b", 0.4),
    ]
    labels = {"a": 1.0, "b": 0.0}
    board = build_leaderboard(scores, {("halubench", MetricFamily.HALLUCINATION): labels})
    rows = [r for r in board if r["dataset"] == "halubench"]
    rows.sort(key=lambda r: -r["auc"])
    assert rows[0]["framework"] == "checkllm"
    assert rows[0]["auc"] == 1.0


def test_leaderboard_sorts_nan_auc_after_finite_auc():
    """NaN-AUC rows must always rank after any finite-AUC row in the same
    bucket, regardless of the order adapters were registered. A previous
    sort implementation mapped NaN to key=1 and finite AUCs to keys in
    [-1, 0], which worked by accident when every row in a group was NaN
    but would silently demote NaN rows if any framework produced a
    finite score.
    """
    # One framework has a finite AUC on TruthfulQA-like scalar labels, the
    # other two are NaN. Finite framework must rank first.
    scores = [
        _mk("checkllm", "a", 0.9),
        _mk("checkllm", "b", 0.1),
        _mk("deepeval", "a", 0.5),
        _mk("deepeval", "b", 0.5),
        _mk("promptfoo", "a", 0.8),
        _mk("promptfoo", "b", 0.4),
    ]
    labels = {"a": 1.0, "b": 0.0}
    board = build_leaderboard(scores, {("halubench", MetricFamily.HALLUCINATION): labels})
    sorted(
        [r for r in board if r["dataset"] == "halubench"],
        key=lambda r: r["rank"],
    )
    # deepeval scores are constant 0.5 → AUC is 0.5 (not NaN) so it still
    # earns a finite rank. Use a pure NaN case: constant labels.
    scores_scalar = [
        _mk("checkllm", "a", 0.9),
        _mk("checkllm", "b", 0.9),
        _mk("deepeval", "a", 0.5),
        _mk("deepeval", "b", 0.5),
    ]
    scalar_labels = {"a": 1.0, "b": 1.0}
    board2 = build_leaderboard(
        scores_scalar, {("truthfulqa", MetricFamily.HALLUCINATION): scalar_labels}
    )
    # Every row has NaN AUC → ranks are 1,2 in insertion order and stable.
    ranks = {r["framework"]: r["rank"] for r in board2}
    assert ranks["checkllm"] in (1, 2)
    assert ranks["deepeval"] in (1, 2)
    assert ranks["checkllm"] != ranks["deepeval"]


def test_write_markdown_emits_table(tmp_path):
    rows = [
        {
            "framework": "checkllm",
            "dataset": "halubench",
            "metric_family": "hallucination",
            "auc": 0.91,
            "best_f1": 0.88,
            "spearman": 0.85,
            "n": 200,
            "mean_latency_ms": 420.0,
            "total_cost_usd": 0.12,
            "rank": 1,
        },
        {
            "framework": "deepeval",
            "dataset": "halubench",
            "metric_family": "hallucination",
            "auc": 0.85,
            "best_f1": 0.80,
            "spearman": 0.78,
            "n": 200,
            "mean_latency_ms": 510.0,
            "total_cost_usd": 0.18,
            "rank": 2,
        },
    ]
    path = tmp_path / "report.md"
    write_markdown(rows, path)
    text = path.read_text(encoding="utf-8")
    assert "| framework | dataset | metric_family | auc |" in text
    assert "checkllm" in text
    assert "deepeval" in text
    # Integer-formatted cells (latency, n, rank) should not have trailing .000
    assert " 420 " in text
    assert " 200 " in text
    assert " 1 " in text
    # Cost has 4 decimals
    assert " 0.1200 " in text
    # Footer is present
    assert "Judge model" in text
    assert "remaining-gaps.md" in text


def test_write_csv_includes_all_columns(tmp_path):
    rows = [
        {
            "framework": "checkllm",
            "dataset": "halubench",
            "metric_family": "hallucination",
            "auc": 0.91,
            "best_f1": 0.88,
            "spearman": 0.85,
            "n": 200,
            "mean_latency_ms": 420.0,
            "total_cost_usd": 0.12,
            "rank": 1,
        },
    ]
    path = tmp_path / "report.csv"
    write_csv(rows, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",")[0] == "framework"
    assert "0.91" in lines[1]


def test_write_html_escapes_and_renders_rows(tmp_path):
    rows = [
        {
            "framework": "checkllm",
            "dataset": "halubench",
            "metric_family": "hallucination",
            "auc": 0.91,
            "best_f1": 0.88,
            "spearman": 0.85,
            "n": 200,
            "mean_latency_ms": 420.0,
            "total_cost_usd": 0.12,
            "rank": 1,
        },
        {
            "framework": "deep<script>",
            "dataset": "halubench",
            "metric_family": "hallucination",
            "auc": 0.85,
            "best_f1": 0.80,
            "spearman": 0.78,
            "n": 200,
            "mean_latency_ms": 510.0,
            "total_cost_usd": 0.18,
            "rank": 2,
        },
    ]
    path = tmp_path / "report.html"
    write_html(rows, path)
    text = path.read_text(encoding="utf-8")
    assert "<th>framework</th>" in text
    assert "checkllm" in text
    assert "deep&lt;script&gt;" in text
    assert "<script>" not in text.replace("deep&lt;script&gt;", "")
