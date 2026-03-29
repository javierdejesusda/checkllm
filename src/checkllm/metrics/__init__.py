from __future__ import annotations

import importlib.metadata
from typing import Any, Callable

from checkllm.metrics.answer_completeness import AnswerCompletenessMetric
from checkllm.metrics.bias import BiasMetric
from checkllm.metrics.consistency import ConsistencyMetric
from checkllm.metrics.context_relevance import ContextRelevanceMetric
from checkllm.metrics.contextual_precision import ContextualPrecisionMetric
from checkllm.metrics.contextual_recall import ContextualRecallMetric
from checkllm.metrics.conversation_completeness import ConversationCompletenessMetric
from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.metrics.g_eval import GEvalMetric
from checkllm.metrics.groundedness import GroundednessMetric
from checkllm.metrics.instruction_following import InstructionFollowingMetric
from checkllm.metrics.knowledge_retention import KnowledgeRetentionMetric
from checkllm.metrics.role_adherence import RoleAdherenceMetric
from checkllm.metrics.summarization import SummarizationMetric
from checkllm.metrics.task_completion import TaskCompletionMetric
from checkllm.metrics.tool_accuracy import ToolAccuracyMetric
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
