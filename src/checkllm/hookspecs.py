"""Hook specifications for the checkllm plugin system.

checkllm uses pluggy to allow third-party packages to extend behavior
at well-defined extension points. Plugin authors implement hook functions
decorated with @hookimpl, then register via the ``checkllm`` entry
point group or by calling ``get_plugin_manager().register(plugin)``.
"""
from __future__ import annotations

from typing import Any

import pluggy

PROJECT_NAME = "checkllm"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class CheckllmHookSpec:
    """Hook specifications for checkllm plugins."""

    @hookspec
    def checkllm_before_check(
        self, metric_name: str, kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Called before a check is run.

        Args:
            metric_name: Name of the check (e.g., "hallucination").
            kwargs: Keyword arguments that will be passed to the check.

        Returns:
            Modified kwargs dict, or None to leave unchanged.
        """

    @hookspec
    def checkllm_after_check(self, result: Any, metric_name: str) -> None:
        """Called after a check completes (pass or fail).

        Args:
            result: The CheckResult object.
            metric_name: Name of the check that ran.
        """

    @hookspec
    def checkllm_on_failure(self, result: Any, metric_name: str) -> None:
        """Called when a check fails.

        Args:
            result: The failed CheckResult object.
            metric_name: Name of the check that failed.
        """

    @hookspec
    def checkllm_modify_judge_prompt(
        self, prompt: str, metric_name: str,
    ) -> str | None:
        """Called before sending a prompt to the LLM judge.

        Args:
            prompt: The prompt about to be sent.
            metric_name: Name of the metric requesting judging.

        Returns:
            Modified prompt string, or None to leave unchanged.
        """

    @hookspec
    def checkllm_register_checks(self, collector: Any) -> None:
        """Called during CheckCollector initialization.

        Args:
            collector: The CheckCollector being initialized.
        """


def get_plugin_manager() -> pluggy.PluginManager:
    """Create and return a configured PluginManager."""
    pm = pluggy.PluginManager(PROJECT_NAME)
    pm.add_hookspecs(CheckllmHookSpec)
    pm.load_setuptools_entrypoints(PROJECT_NAME)
    return pm


_plugin_manager: pluggy.PluginManager | None = None


def plugin_manager() -> pluggy.PluginManager:
    """Return the global plugin manager (created on first call)."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = get_plugin_manager()
    return _plugin_manager
