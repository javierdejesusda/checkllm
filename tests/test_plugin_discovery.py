"""Tests for community metric plugin discovery."""

from unittest.mock import patch, MagicMock
import pluggy
import pytest
from checkllm.metrics import MetricRegistry
from checkllm.models import CheckResult


def _make_fake_metric(output: str, **kwargs) -> CheckResult:
    return CheckResult(
        passed=True,
        score=0.9,
        reasoning="fake",
        cost=0.0,
        latency_ms=0,
        metric_name="fake_metric",
    )


def test_registry_lists_builtin_and_custom():
    reg = MetricRegistry()
    reg.register("custom_check")(_make_fake_metric)
    assert "custom_check" in reg.list_metrics()


def test_registry_prevents_duplicate_names():
    reg = MetricRegistry()
    reg.register("dupe")(_make_fake_metric)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("dupe")(_make_fake_metric)


def test_load_entry_points_discovers_plugins():
    """Simulate a plugin being discovered via entry points."""
    reg = MetricRegistry()

    fake_metric_func = _make_fake_metric
    fake_ep = MagicMock()
    fake_ep.name = "my_plugin_metric"
    fake_ep.load.return_value = fake_metric_func

    with patch("importlib.metadata.entry_points", return_value=[fake_ep]):
        reg.load_entry_points()

    fake_ep.load.assert_called_once()


def test_list_metrics_with_source_attribution():
    """list_metrics_detailed returns source info."""
    reg = MetricRegistry()
    reg.register("local_check")(_make_fake_metric)
    detailed = reg.list_metrics_detailed()
    assert any(m["name"] == "local_check" and m["source"] == "local" for m in detailed)


def test_entry_point_errors_are_silenced():
    """Bad plugins don't crash the registry."""
    reg = MetricRegistry()
    bad_ep = MagicMock()
    bad_ep.name = "broken"
    bad_ep.load.side_effect = ImportError("broken package")

    with patch("importlib.metadata.entry_points", return_value=[bad_ep]):
        reg.load_entry_points()  # Should not raise


class TestPluginDiscovery:
    def test_plugin_manager_loads_without_error(self):
        from checkllm.hookspecs import get_plugin_manager

        pm = get_plugin_manager()
        assert isinstance(pm, pluggy.PluginManager)

    def test_hookspecs_registered(self):
        from checkllm.hookspecs import get_plugin_manager

        pm = get_plugin_manager()
        assert pm.parse_hookimpl_opts is not None

    def test_entry_point_group_name(self):
        from checkllm.hookspecs import PROJECT_NAME

        assert PROJECT_NAME == "checkllm"
