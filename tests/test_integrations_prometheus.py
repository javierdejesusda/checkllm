"""Tests for the Prometheus / Grafana exporter.

The tests skip automatically when ``prometheus_client`` is not available
in the environment, since installation of the package is required for
the exporter to do anything meaningful.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from checkllm.models import CheckResult

prometheus_client = pytest.importorskip(
    "prometheus_client",
    reason="prometheus_client is required for these tests",
)


def _make_check(
    metric: str = "relevance",
    passed: bool = True,
    cost: float = 0.01,
    latency_ms: int = 500,
) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reasoning="ok" if passed else "nope",
        cost=cost,
        latency_ms=latency_ms,
        metric_name=metric,
    )


def test_metrics_register_on_isolated_registry():
    from prometheus_client import CollectorRegistry

    from checkllm.integrations.prometheus import PrometheusExporter

    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry=registry)

    assert exporter.registry is registry
    text = exporter.generate_latest().decode()
    assert "checkllm_evaluations_total" in text
    assert "checkllm_evaluation_duration_seconds" in text
    assert "checkllm_judge_cost_usd_total" in text
    assert "checkllm_judge_tokens_total" in text


def test_record_check_updates_counters():
    from prometheus_client import CollectorRegistry

    from checkllm.integrations.prometheus import PrometheusExporter

    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry=registry)

    exporter.record_check(
        _make_check(passed=True, cost=0.05, latency_ms=1_000),
        judge="gpt-4o",
        prompt_tokens=100,
        completion_tokens=25,
    )
    exporter.record_check(
        _make_check(passed=False, cost=0.02, latency_ms=200),
        judge="gpt-4o",
    )

    passed_samples = registry.get_sample_value(
        "checkllm_evaluations_total",
        labels={"metric": "relevance", "status": "passed"},
    )
    failed_samples = registry.get_sample_value(
        "checkllm_evaluations_total",
        labels={"metric": "relevance", "status": "failed"},
    )
    cost_samples = registry.get_sample_value(
        "checkllm_judge_cost_usd_total",
        labels={"metric": "relevance", "judge": "gpt-4o"},
    )
    prompt_tokens = registry.get_sample_value(
        "checkllm_judge_tokens_total",
        labels={
            "metric": "relevance",
            "judge": "gpt-4o",
            "kind": "prompt",
        },
    )
    completion_tokens = registry.get_sample_value(
        "checkllm_judge_tokens_total",
        labels={
            "metric": "relevance",
            "judge": "gpt-4o",
            "kind": "completion",
        },
    )

    assert passed_samples == 1.0
    assert failed_samples == 1.0
    assert cost_samples == pytest.approx(0.07)
    assert prompt_tokens == 100.0
    assert completion_tokens == 25.0


def test_duration_histogram_observes_in_seconds():
    from prometheus_client import CollectorRegistry

    from checkllm.integrations.prometheus import PrometheusExporter

    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry=registry)

    exporter.record_check(
        _make_check(latency_ms=2_500), judge="gpt-4o"
    )

    count = registry.get_sample_value(
        "checkllm_evaluation_duration_seconds_count",
        labels={"metric": "relevance"},
    )
    total = registry.get_sample_value(
        "checkllm_evaluation_duration_seconds_sum",
        labels={"metric": "relevance"},
    )

    assert count == 1.0
    assert total == pytest.approx(2.5)


def test_custom_namespace_prefixes_metric_names():
    from prometheus_client import CollectorRegistry

    from checkllm.integrations.prometheus import PrometheusExporter

    registry = CollectorRegistry()
    exporter = PrometheusExporter(namespace="myapp", registry=registry)

    text = exporter.generate_latest().decode()
    assert "myapp_evaluations_total" in text
    assert "myapp_evaluation_duration_seconds" in text


def test_start_http_server_calls_library():
    from checkllm.integrations.prometheus import PrometheusExporter

    exporter = PrometheusExporter(port=0)
    fake = MagicMock(return_value=MagicMock())
    exporter._prom.start_http_server = fake

    exporter.start_http_server(port=12345)
    fake.assert_called_once_with(12345, registry=exporter.registry)


def test_import_error_when_sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    from checkllm.integrations.prometheus import PrometheusExporter

    with pytest.raises(ImportError, match="prometheus-client"):
        PrometheusExporter()


def test_factory_returns_prometheus_exporter():
    from checkllm.integrations import get_tracer
    from checkllm.integrations.prometheus import PrometheusExporter

    exporter = get_tracer("prometheus")
    assert isinstance(exporter, PrometheusExporter)


def test_get_tracer_unknown_name_raises():
    from checkllm.integrations import get_tracer

    with pytest.raises(ValueError, match="Unknown tracer"):
        get_tracer("does-not-exist")
