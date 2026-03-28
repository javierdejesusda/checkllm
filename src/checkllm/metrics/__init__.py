from __future__ import annotations

import importlib.metadata
from typing import Any, Callable

from checkllm.models import CheckResult


class MetricRegistry:
    """Registry for custom metric functions."""

    def __init__(self) -> None:
        self.metrics: dict[str, Callable[..., CheckResult]] = {}

    def register(self, name: str) -> Callable:
        """Decorator to register a custom metric function."""

        def decorator(func: Callable[..., CheckResult]) -> Callable[..., CheckResult]:
            if name in self.metrics:
                raise ValueError(
                    f"Metric '{name}' is already registered. "
                    "Choose a different name."
                )
            self.metrics[name] = func
            return func

        return decorator

    def list_metrics(self) -> list[str]:
        return list(self.metrics.keys())

    def load_entry_points(self) -> None:
        """Discover and load plugins from checkllm.metrics entry points."""
        try:
            eps = importlib.metadata.entry_points(group="checkllm.metrics")
        except TypeError:
            eps = importlib.metadata.entry_points().get("checkllm.metrics", [])
        for ep in eps:
            try:
                register_func = ep.load()
                if callable(register_func):
                    register_func(self)
            except Exception:
                pass


_global_registry = MetricRegistry()


def metric(name: str) -> Callable:
    """Global decorator to register a custom metric."""
    return _global_registry.register(name)
