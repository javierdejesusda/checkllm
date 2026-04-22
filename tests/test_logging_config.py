"""Tests for structured logging configuration."""

import logging

from checkllm.logging_config import setup_logging


class TestLoggingSetup:
    def test_default_level(self):
        setup_logging()
        logger = logging.getLogger("checkllm")
        assert logger.level == logging.WARNING

    def test_custom_level(self):
        setup_logging("DEBUG")
        logger = logging.getLogger("checkllm")
        assert logger.level == logging.DEBUG

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CHECKLLM_LOG_LEVEL", "INFO")
        setup_logging()
        logger = logging.getLogger("checkllm")
        assert logger.level == logging.INFO

    def test_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CHECKLLM_LOG_LEVEL", "INFO")
        setup_logging("ERROR")
        logger = logging.getLogger("checkllm")
        assert logger.level == logging.ERROR

    def test_child_loggers_inherit(self):
        setup_logging("DEBUG")
        cache_logger = logging.getLogger("checkllm.cache")
        assert cache_logger.getEffectiveLevel() == logging.DEBUG
