from bench.scoring import roc_auc, best_f1, spearman, summarize_scores
from bench.schema import BenchmarkScore, MetricFamily


def _mk(sample_id: str, score: float) -> BenchmarkScore:
    return BenchmarkScore(
        framework="checkllm",
        dataset="halubench",
        metric_family=MetricFamily.HALLUCINATION,
        metric_name="hallucination",
        sample_id=sample_id,
        score=score,
        passed=score >= 0.5,
        latency_ms=100,
        cost_usd=0.0,
        judge_model="mock",
    )


def test_perfect_auc_when_scores_match_labels():
    scores = [_mk("a", 0.9), _mk("b", 0.1), _mk("c", 0.8), _mk("d", 0.2)]
    labels = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    assert roc_auc(scores, labels) == 1.0


def test_best_f1_returns_f1_and_threshold():
    scores = [_mk("a", 0.9), _mk("b", 0.1), _mk("c", 0.8), _mk("d", 0.2)]
    labels = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    f1, threshold = best_f1(scores, labels)
    assert f1 == 1.0
    assert 0.2 < threshold <= 0.9


def test_spearman_is_one_for_monotonic():
    scores = [_mk("a", 0.9), _mk("b", 0.6), _mk("c", 0.3)]
    labels = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert spearman(scores, labels) == 1.0


def test_summarize_scores_returns_dict_with_known_keys():
    scores = [_mk("a", 0.9), _mk("b", 0.1), _mk("c", 0.8), _mk("d", 0.2)]
    labels = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    summary = summarize_scores(scores, labels)
    assert set(summary.keys()) >= {
        "auc",
        "best_f1",
        "best_threshold",
        "n",
        "mean_latency_ms",
        "total_cost_usd",
    }
    assert summary["n"] == 4
