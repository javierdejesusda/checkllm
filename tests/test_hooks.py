"""Verify the checkllm hook system works with pluggy."""
import pluggy
import pytest

from checkllm.hookspecs import CheckllmHookSpec, hookimpl
from checkllm.models import CheckResult


class TestHookSystem:
    def test_hookspec_exists(self):
        assert hasattr(CheckllmHookSpec, "checkllm_before_check")
        assert hasattr(CheckllmHookSpec, "checkllm_after_check")
        assert hasattr(CheckllmHookSpec, "checkllm_on_failure")
        assert hasattr(CheckllmHookSpec, "checkllm_modify_judge_prompt")
        assert hasattr(CheckllmHookSpec, "checkllm_register_checks")

    def test_plugin_manager_creation(self):
        from checkllm.hookspecs import get_plugin_manager
        pm = get_plugin_manager()
        assert isinstance(pm, pluggy.PluginManager)

    def test_hookimpl_decorator(self):
        @hookimpl
        def checkllm_after_check(result, metric_name):
            pass
        assert hasattr(checkllm_after_check, "checkllm_impl")

    def test_plugin_registration_and_call(self):
        from checkllm.hookspecs import get_plugin_manager

        call_log = []

        class MyPlugin:
            @hookimpl
            def checkllm_after_check(self, result, metric_name):
                call_log.append(metric_name)

        pm = get_plugin_manager()
        pm.register(MyPlugin())

        result = CheckResult(
            passed=True, score=1.0, reasoning="OK",
            cost=0.0, latency_ms=0, metric_name="contains",
        )
        pm.hook.checkllm_after_check(result=result, metric_name="contains")
        assert call_log == ["contains"]
