from checkllm.dashboard import (
    AlertConfig,
    AlertEvent,
    ComparisonView,
    build_comparison_view,
    check_alerts,
)
from checkllm.models import CheckResult


class TestComparisonView:
    def test_model(self):
        cv = ComparisonView(
            snapshot_a="baseline",
            snapshot_b="current",
            metrics_diff={"relevance": 0.1, "toxicity": -0.05},
            improved=["relevance"],
            regressed=["toxicity"],
        )
        assert len(cv.improved) == 1
        assert len(cv.regressed) == 1

    def test_build_comparison_view(self):
        results_a = {
            "test_1": [
                CheckResult(
                    passed=True,
                    score=0.8,
                    reasoning="OK",
                    cost=0.0,
                    latency_ms=0,
                    metric_name="relevance",
                ),
            ],
        }
        results_b = {
            "test_1": [
                CheckResult(
                    passed=True,
                    score=0.9,
                    reasoning="Better",
                    cost=0.0,
                    latency_ms=0,
                    metric_name="relevance",
                ),
            ],
        }
        cv = build_comparison_view(results_a, results_b)
        assert "relevance" in cv.improved or cv.metrics_diff.get("relevance", 0) > 0

    def test_empty_results(self):
        cv = build_comparison_view({}, {})
        assert cv.metrics_diff == {}


class TestAlertConfig:
    def test_defaults(self):
        config = AlertConfig()
        assert not config.enabled
        assert config.min_score_threshold == 0.5

    def test_check_alerts_on_failure(self):
        config = AlertConfig(enabled=True, alert_on_failure=True)
        results = {
            "test_1": [
                CheckResult(
                    passed=False,
                    score=0.3,
                    reasoning="Bad",
                    cost=0.0,
                    latency_ms=0,
                    metric_name="toxicity",
                ),
            ],
        }
        events = check_alerts(results, config)
        assert len(events) >= 1
        assert any(e.event_type == "failure" for e in events)

    def test_check_alerts_threshold_breach(self):
        config = AlertConfig(enabled=True, min_score_threshold=0.7)
        results = {
            "test_1": [
                CheckResult(
                    passed=True,
                    score=0.5,
                    reasoning="OK",
                    cost=0.0,
                    latency_ms=0,
                    metric_name="relevance",
                ),
            ],
        }
        events = check_alerts(results, config)
        assert any(e.event_type == "threshold_breach" for e in events)

    def test_no_alerts_when_all_pass(self):
        config = AlertConfig(enabled=True, min_score_threshold=0.5)
        results = {
            "test_1": [
                CheckResult(
                    passed=True,
                    score=0.9,
                    reasoning="Great",
                    cost=0.0,
                    latency_ms=0,
                    metric_name="relevance",
                ),
            ],
        }
        events = check_alerts(results, config)
        assert len(events) == 0

    def test_alert_event_model(self):
        e = AlertEvent(
            event_type="regression",
            metric_name="relevance",
            details="Score dropped",
            score=0.6,
            threshold=0.8,
        )
        assert e.event_type == "regression"
