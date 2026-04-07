"""Integration test: verify all maturity improvements work together."""
import warnings


from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector
from checkllm.deprecations import CheckllmDeprecationWarning, deprecated
from checkllm.hookspecs import hookimpl, plugin_manager


class TestMaturityIntegration:
    def test_deterministic_check_with_rich_result(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.contains("The answer is 42", "42")
        assert result.passed
        assert result.input_preview is not None

    def test_failed_check_has_format_failure(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.contains("hello", "goodbye")
        assert not result.passed
        text = result.format_failure()
        assert "contains" in text
        assert "Score:" in text

    def test_hook_fires_during_check(self):
        log = []

        class Logger:
            @hookimpl
            def checkllm_after_check(self, result, metric_name):
                log.append(metric_name)

        pm = plugin_manager()
        listener = Logger()
        pm.register(listener)
        try:
            config = CheckllmConfig()
            c = CheckCollector(config=config)
            c.contains("hello", "hello")
            c.regex("abc123", r"\d+")
            assert log == ["contains", "regex"]
        finally:
            pm.unregister(listener)

    def test_deprecation_warning_works(self):
        @deprecated("Use new_api()", removal_version="5.0")
        def old_api():
            return True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert old_api()
            assert len(w) == 1
            assert issubclass(w[0].category, CheckllmDeprecationWarning)

    def test_check_collector_has_both_mixins(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        assert hasattr(c, "contains")
        assert hasattr(c, "bleu")
        assert hasattr(c, "hallucination")
        assert hasattr(c, "relevance")
        assert hasattr(c, "that")
        assert hasattr(c, "expect")
