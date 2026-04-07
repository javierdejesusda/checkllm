"""Verify hooks fire during actual check execution."""
import pytest

from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector
from checkllm.hookspecs import hookimpl, plugin_manager


class TestHooksIntegration:
    def test_after_check_fires_on_deterministic(self):
        log = []

        class Listener:
            @hookimpl
            def checkllm_after_check(self, result, metric_name):
                log.append(("after", metric_name, result.passed))

        pm = plugin_manager()
        listener = Listener()
        pm.register(listener)
        try:
            config = CheckllmConfig()
            c = CheckCollector(config=config)
            c.contains("hello", "hello")
            assert log == [("after", "contains", True)]
        finally:
            pm.unregister(listener)

    def test_on_failure_fires_only_on_fail(self):
        log = []

        class FailListener:
            @hookimpl
            def checkllm_on_failure(self, result, metric_name):
                log.append(metric_name)

        pm = plugin_manager()
        listener = FailListener()
        pm.register(listener)
        try:
            config = CheckllmConfig()
            c = CheckCollector(config=config)
            c.contains("hello", "hello")
            c.contains("hello", "xyz")
            assert log == ["contains"]
        finally:
            pm.unregister(listener)

    def test_before_check_can_see_metric_name(self):
        log = []

        class BeforeListener:
            @hookimpl
            def checkllm_before_check(self, metric_name, kwargs):
                log.append(metric_name)
                return None

        pm = plugin_manager()
        listener = BeforeListener()
        pm.register(listener)
        try:
            config = CheckllmConfig()
            c = CheckCollector(config=config)
            c.contains("hello", "hello")
            assert "contains" in log
        finally:
            pm.unregister(listener)
