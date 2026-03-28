"""Structured logging configuration for checkllm."""
from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure the checkllm logger hierarchy.

    Level can be set via:
      1. ``level`` argument
      2. ``CHECKLLM_LOG_LEVEL`` environment variable
      3. Defaults to WARNING
    """
    resolved = level or os.environ.get("CHECKLLM_LOG_LEVEL", "WARNING")
    numeric = getattr(logging, resolved.upper(), logging.WARNING)

    root_logger = logging.getLogger("checkllm")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(name)s] %(levelname)s: %(message)s"
            )
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric)
