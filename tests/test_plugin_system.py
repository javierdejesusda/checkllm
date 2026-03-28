import pytest

from checkllm.metrics import MetricRegistry, metric
from checkllm.models import CheckResult


class TestMetricDecorator:
    def test_register_custom_metric(self):
        registry = MetricRegistry()

        @registry.register("custom_test")
        def my_metric(output: str, **kwargs) -> CheckResult:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="custom pass",
                cost=0.0,
                latency_ms=0,
                metric_name="custom_test",
            )

        assert "custom_test" in registry.metrics
        result = registry.metrics["custom_test"]("test output")
        assert result.passed is True
        assert result.metric_name == "custom_test"

    def test_register_duplicate_raises(self):
        registry = MetricRegistry()

        @registry.register("dup")
        def metric_a(output: str, **kwargs) -> CheckResult:
            return CheckResult(
                passed=True, score=1.0, reasoning="", cost=0.0,
                latency_ms=0, metric_name="dup",
            )

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("dup")
            def metric_b(output: str, **kwargs) -> CheckResult:
                return CheckResult(
                    passed=True, score=1.0, reasoning="", cost=0.0,
                    latency_ms=0, metric_name="dup",
                )

    def test_list_registered_metrics(self):
        registry = MetricRegistry()

        @registry.register("alpha")
        def m1(output: str, **kwargs) -> CheckResult:
            return CheckResult(
                passed=True, score=1.0, reasoning="", cost=0.0,
                latency_ms=0, metric_name="alpha",
            )

        @registry.register("beta")
        def m2(output: str, **kwargs) -> CheckResult:
            return CheckResult(
                passed=True, score=1.0, reasoning="", cost=0.0,
                latency_ms=0, metric_name="beta",
            )

        assert set(registry.list_metrics()) == {"alpha", "beta"}


class TestGlobalMetricDecorator:
    def test_global_metric_decorator(self):
        @metric("global_test_metric")
        def my_global(output: str, **kwargs) -> CheckResult:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="global pass",
                cost=0.0,
                latency_ms=0,
                metric_name="global_test_metric",
            )

        from checkllm.metrics import _global_registry

        assert "global_test_metric" in _global_registry.metrics
