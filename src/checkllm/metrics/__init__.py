from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, Callable

from checkllm.metrics.answer_completeness import AnswerCompletenessMetric
from checkllm.metrics.bias import BiasMetric
from checkllm.metrics.citation_accuracy import CitationAccuracyMetric
from checkllm.metrics.code_correctness import CodeCorrectnessMetric
from checkllm.metrics.comparative_quality import ComparativeQualityMetric
from checkllm.metrics.consistency import ConsistencyMetric
from checkllm.metrics.context_entity_recall import ContextEntityRecallMetric
from checkllm.metrics.context_relevance import ContextRelevanceMetric
from checkllm.metrics.contextual_precision import ContextualPrecisionMetric
from checkllm.metrics.contextual_recall import ContextualRecallMetric
from checkllm.metrics.conversation_completeness import ConversationCompletenessMetric
from checkllm.metrics.datacompy_score import DataCompyMetric
from checkllm.metrics.dual_judge_nv import (
    NVAnswerAccuracyMetric,
    NVContextRelevanceMetric,
    NVResponseGroundednessMetric,
)
from checkllm.metrics.factual_correctness import FactualCorrectnessMetric
from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.metrics.faithfulness_hhem import FaithfulnessHHEMMetric
from checkllm.metrics.g_eval import GEvalMetric
from checkllm.metrics.groundedness import GroundednessMetric
from checkllm.metrics.chart_value_extraction import ChartValueExtractionMetric
from checkllm.metrics.diagram_comprehension import DiagramComprehensionMetric
from checkllm.metrics.image_captioning_quality import ImageCaptioningQualityMetric
from checkllm.metrics.image_consistency import ImageConsistencyMetric
from checkllm.metrics.image_editing import ImageEditingMetric
from checkllm.metrics.image_reference import ImageReferenceMetric
from checkllm.metrics.image_safety import ImageSafetyMetric
from checkllm.metrics.image_text_alignment import ImageTextAlignmentMetric
from checkllm.metrics.ocr_accuracy import OCRAccuracyMetric
from checkllm.metrics.visual_faithfulness import VisualFaithfulnessMetric
from checkllm.metrics.visual_hallucination import VisualHallucinationMetric
from checkllm.metrics.visual_reasoning import VisualReasoningMetric
from checkllm.metrics.instruction_completeness import InstructionCompletenessMetric
from checkllm.metrics.kb_faithfulness import KBFaithfulnessMetric
from checkllm.metrics.instruction_following import InstructionFollowingMetric
from checkllm.metrics.knowledge_retention import KnowledgeRetentionMetric
from checkllm.metrics.noise_sensitivity import NoiseSensitivityMetric
from checkllm.metrics.nonllm_context_precision import NonLLMContextPrecisionMetric
from checkllm.metrics.nonllm_context_recall import NonLLMContextRecallMetric
from checkllm.metrics.per_turn import (
    TurnCoherenceMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from checkllm.metrics.prompt_alignment import PromptAlignmentMetric
from checkllm.metrics.quoted_spans import QuotedSpansAlignmentMetric
from checkllm.metrics.response_completeness import ResponseCompletenessMetric
from checkllm.metrics.role_adherence import RoleAdherenceMetric
from checkllm.metrics.sql_equivalence import SQLEquivalenceMetric
from checkllm.metrics.summarization import SummarizationMetric
from checkllm.metrics.task_completion import TaskCompletionMetric
from checkllm.metrics.tool_accuracy import ToolAccuracyMetric
from checkllm.metrics.tool_call_f1 import ToolCallF1Metric
from checkllm.metrics.topic_adherence import TopicAdherenceMetric
from checkllm.metrics.trajectory import (
    TrajectoryGoalSuccessMetric,
    TrajectoryStepCountMetric,
    TrajectoryToolArgsMatchMetric,
    TrajectoryToolSequenceMetric,
)
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.metrics")


class MetricRegistry:
    """Registry for custom metric functions."""

    def __init__(self) -> None:
        self.metrics: dict[str, Callable[..., CheckResult]] = {}
        self._sources: dict[str, str] = {}

    def register(self, name: str, source: str = "local") -> Callable:
        """Decorator to register a custom metric function."""

        def decorator(func: Callable[..., CheckResult]) -> Callable[..., CheckResult]:
            if name in self.metrics:
                raise ValueError(f"Metric '{name}' is already registered. Choose a different name.")
            self.metrics[name] = func
            self._sources[name] = source
            return func

        return decorator

    def list_metrics(self) -> list[str]:
        return list(self.metrics.keys())

    def list_metrics_detailed(self) -> list[dict[str, str]]:
        """Return metrics with source attribution."""
        return [
            {"name": name, "source": self._sources.get(name, "unknown")} for name in self.metrics
        ]

    def load_entry_points(self) -> None:
        """Discover and load plugins from checkllm.metrics entry points."""
        try:
            eps = importlib.metadata.entry_points(group="checkllm.metrics")
        except TypeError:
            eps = importlib.metadata.entry_points().get("checkllm.metrics", [])
        for ep in eps:
            try:
                loaded = ep.load()
                if callable(loaded) and ep.name not in self.metrics:
                    self.metrics[ep.name] = loaded
                    self._sources[ep.name] = f"plugin:{ep.dist.name if ep.dist else 'unknown'}"
                    logger.info("Loaded plugin metric: %s", ep.name)
            except Exception as exc:
                logger.debug("Failed to load metric plugin %s: %s", ep.name, exc)


_global_registry = MetricRegistry()


def metric(name: str) -> Callable:
    """Global decorator to register a custom metric."""
    return _global_registry.register(name)
