"""Mixin providing LLM-as-judge check methods for CheckCollector."""
from __future__ import annotations

from typing import Any

from checkllm.cache import _cache_key
from checkllm.models import CheckResult


class JudgeChecksMixin:
    """LLM judge check methods (sync and async) for CheckCollector."""

    def hallucination(
        self,
        output: str,
        context: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.hallucination import HallucinationMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="hallucination",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context),
            cache_kwargs={"output": output, "context": context, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def relevance(
        self,
        output: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.relevance import RelevanceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="relevance",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, query=query),
            cache_kwargs={"output": output, "query": query, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def toxicity(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.toxicity import ToxicityMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="toxicity",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def rubric(
        self,
        output: str,
        criteria: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.rubric import RubricMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="rubric",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, criteria=criteria, threshold=t),
            cache_kwargs={"output": output, "criteria": criteria, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def fluency(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.fluency import FluencyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = FluencyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="fluency",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def coherence(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.coherence import CoherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CoherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="coherence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def sentiment(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.sentiment import SentimentMetric
        t = threshold if threshold is not None else 0.5  # neutral default
        metric = SentimentMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="sentiment",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def correctness(
        self,
        output: str,
        expected: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.correctness import CorrectnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CorrectnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="correctness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, expected=expected),
            cache_kwargs={"output": output, "expected": expected, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def faithfulness(
        self,
        output: str,
        context: str,
        query: str | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.faithfulness import FaithfulnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = FaithfulnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "context": context, "threshold": str(t)}
        if query:
            cache_kw["query"] = query
        return self._cached_judge_check(
            metric_name="faithfulness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context, query=query),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def context_relevance(
        self,
        context: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.context_relevance import ContextRelevanceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ContextRelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="context_relevance",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(context=context, query=query),
            cache_kwargs={"context": context, "query": query, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=context,
        )

    def answer_completeness(
        self,
        output: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.answer_completeness import AnswerCompletenessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = AnswerCompletenessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="answer_completeness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, query=query),
            cache_kwargs={"output": output, "query": query, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def instruction_following(
        self,
        output: str,
        instructions: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.instruction_following import InstructionFollowingMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = InstructionFollowingMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="instruction_following",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, instructions=instructions),
            cache_kwargs={"output": output, "instructions": instructions, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def summarization(
        self,
        output: str,
        source: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.summarization import SummarizationMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = SummarizationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="summarization",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, source=source),
            cache_kwargs={"output": output, "source": source, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def bias(
        self,
        output: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.bias import BiasMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = BiasMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "threshold": str(t)}
        if categories:
            cache_kw["categories"] = ",".join(categories)
        return self._cached_judge_check(
            metric_name="bias",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, categories=categories),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def consistency(
        self,
        outputs: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.consistency import ConsistencyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ConsistencyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="consistency",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(outputs=outputs),
            cache_kwargs={"outputs": "|".join(outputs), "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=outputs[0] if outputs else None,
        )

    def groundedness(
        self,
        output: str,
        sources: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.groundedness import GroundednessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = GroundednessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="groundedness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, sources=sources),
            cache_kwargs={"output": output, "sources": "|".join(sources), "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def g_eval(
        self,
        output: str,
        criteria: str,
        steps: list[str] | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.g_eval import GEvalMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = GEvalMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "criteria": criteria, "threshold": str(t)}
        if steps:
            cache_kw["steps"] = "|".join(steps)
        return self._cached_judge_check(
            metric_name="g_eval",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, criteria=criteria, steps=steps),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def contextual_precision(
        self,
        output: str,
        context: list[str],
        query: str,
        expected: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.contextual_precision import ContextualPrecisionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ContextualPrecisionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="contextual_precision",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context, query=query, expected=expected),
            cache_kwargs={"output": output, "context": "|".join(context), "query": query, "expected": expected, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def contextual_recall(
        self,
        output: str,
        context: list[str],
        expected: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.contextual_recall import ContextualRecallMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ContextualRecallMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="contextual_recall",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context, expected=expected),
            cache_kwargs={"output": output, "context": "|".join(context), "expected": expected, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def task_completion(
        self,
        output: str,
        task_description: str,
        constraints: list[str] | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.task_completion import TaskCompletionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = TaskCompletionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "task_description": task_description, "threshold": str(t)}
        if constraints:
            cache_kw["constraints"] = "|".join(constraints)
        return self._cached_judge_check(
            metric_name="task_completion",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, task_description=task_description, constraints=constraints),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def role_adherence(
        self,
        output: str,
        role_description: str,
        query: str | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.role_adherence import RoleAdherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RoleAdherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "role_description": role_description, "threshold": str(t)}
        if query:
            cache_kw["query"] = query
        return self._cached_judge_check(
            metric_name="role_adherence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, role_description=role_description, query=query),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def tool_accuracy(
        self,
        output: str,
        expected_tools: list[dict[str, object]],
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.tool_accuracy import ToolAccuracyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToolAccuracyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        import json as _json
        return self._cached_judge_check(
            metric_name="tool_accuracy",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, expected_tools=expected_tools, query=query),
            cache_kwargs={"output": output, "expected_tools": _json.dumps(expected_tools, sort_keys=True), "query": query, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def knowledge_retention(
        self,
        conversation: Any,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.knowledge_retention import KnowledgeRetentionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = KnowledgeRetentionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        transcript = conversation.format_transcript()
        return self._cached_judge_check(
            metric_name="knowledge_retention",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(conversation=conversation),
            cache_kwargs={"transcript": transcript, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=conversation.format_transcript()[:200],
        )

    def conversation_completeness(
        self,
        conversation: Any,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.conversation_completeness import ConversationCompletenessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ConversationCompletenessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        transcript = conversation.format_transcript()
        return self._cached_judge_check(
            metric_name="conversation_completeness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(conversation=conversation),
            cache_kwargs={"transcript": transcript, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=conversation.format_transcript()[:200],
        )

    def plan_quality(
        self,
        plan: str,
        task: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.plan_quality import PlanQualityMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = PlanQualityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="plan_quality",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(plan=plan, task=task),
            cache_kwargs={"plan": plan, "task": task, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=plan,
        )

    def goal_accuracy(
        self,
        output: str,
        goal: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.goal_accuracy import GoalAccuracyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = GoalAccuracyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="goal_accuracy",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, goal=goal),
            cache_kwargs={"output": output, "goal": goal, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def step_efficiency(
        self,
        steps: list[str],
        task: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.step_efficiency import StepEfficiencyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = StepEfficiencyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="step_efficiency",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(steps=steps, task=task),
            cache_kwargs={"steps": "|".join(steps), "task": task, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=steps[0] if steps else None,
        )

    def argument_correctness(
        self,
        tool_calls: str,
        expected_calls: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.argument_correctness import ArgumentCorrectnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ArgumentCorrectnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="argument_correctness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(tool_calls=tool_calls, expected_calls=expected_calls),
            cache_kwargs={"tool_calls": tool_calls, "expected_calls": expected_calls, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=tool_calls,
        )

    def plan_adherence(
        self,
        plan: str,
        execution_trace: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.plan_adherence import PlanAdherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = PlanAdherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="plan_adherence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(plan=plan, execution_trace=execution_trace),
            cache_kwargs={"plan": plan, "execution_trace": execution_trace, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=plan,
        )

    def pii_detection(
        self,
        output: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.pii_detection import PIIDetectionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = PIIDetectionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="pii_detection",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output),
            cache_kwargs={"output": output, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def misuse_detection(
        self,
        output: str,
        intended_scope: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.misuse_detection import MisuseDetectionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = MisuseDetectionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="misuse_detection",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, intended_scope=intended_scope),
            cache_kwargs={"output": output, "intended_scope": intended_scope, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def role_violation(
        self,
        output: str,
        role_description: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.role_violation import RoleViolationMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RoleViolationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="role_violation",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, role_description=role_description),
            cache_kwargs={"output": output, "role_description": role_description, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def non_advice(
        self,
        output: str,
        restricted_domains: list[str] | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.non_advice import NonAdviceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = NonAdviceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "threshold": str(t)}
        if restricted_domains:
            cache_kw["restricted_domains"] = ",".join(restricted_domains)
        return self._cached_judge_check(
            metric_name="non_advice",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, restricted_domains=restricted_domains),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def image_coherence(
        self,
        image_description: str,
        text_context: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.image_coherence import ImageCoherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ImageCoherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="image_coherence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                image_description=image_description, text_context=text_context
            ),
            cache_kwargs={
                "image_description": image_description,
                "text_context": text_context,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=image_description,
        )

    def image_helpfulness(
        self,
        image_description: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.image_helpfulness import ImageHelpfulnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ImageHelpfulnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="image_helpfulness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                image_description=image_description, query=query
            ),
            cache_kwargs={
                "image_description": image_description,
                "query": query,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=image_description,
        )

    def image_relevance(
        self,
        image_description: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.image_relevance import ImageRelevanceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ImageRelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="image_relevance",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                image_description=image_description, query=query
            ),
            cache_kwargs={
                "image_description": image_description,
                "query": query,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=image_description,
        )

    def multimodal_faithfulness(
        self,
        image_description: str,
        text_output: str,
        source_context: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.multimodal_faithfulness import MultimodalFaithfulnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = MultimodalFaithfulnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="multimodal_faithfulness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                image_description=image_description,
                text_output=text_output,
                source_context=source_context,
            ),
            cache_kwargs={
                "image_description": image_description,
                "text_output": text_output,
                "source_context": source_context,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=image_description,
        )

    def text_to_image(
        self,
        image_description: str,
        original_prompt: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.text_to_image import TextToImageMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = TextToImageMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="text_to_image",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                image_description=image_description, original_prompt=original_prompt
            ),
            cache_kwargs={
                "image_description": image_description,
                "original_prompt": original_prompt,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=image_description,
        )

    def mcp_task_completion(
        self,
        output: str,
        task: str,
        tools_used: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.mcp_task_completion import MCPTaskCompletionMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = MCPTaskCompletionMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="mcp_task_completion",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                output=output, task=task, tools_used=tools_used
            ),
            cache_kwargs={
                "output": output,
                "task": task,
                "tools_used": ",".join(tools_used),
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def mcp_use(
        self,
        output: str,
        tools_available: list[str],
        tools_used: list[str],
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.mcp_use import MCPUseMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = MCPUseMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="mcp_use",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                output=output,
                tools_available=tools_available,
                tools_used=tools_used,
                query=query,
            ),
            cache_kwargs={
                "output": output,
                "tools_available": ",".join(tools_available),
                "tools_used": ",".join(tools_used),
                "query": query,
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def multi_turn_mcp_use(
        self,
        conversation_trace: str,
        tools_available: list[str],
        tools_used: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.multi_turn_mcp_use import MultiTurnMCPUseMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = MultiTurnMCPUseMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="multi_turn_mcp_use",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(
                conversation_trace=conversation_trace,
                tools_available=tools_available,
                tools_used=tools_used,
            ),
            cache_kwargs={
                "conversation_trace": conversation_trace,
                "tools_available": ",".join(tools_available),
                "tools_used": ",".join(tools_used),
                "threshold": str(t),
            },
            runs=runs,
            threshold=t,
            input_preview=conversation_trace,
        )

    def factual_correctness(
        self,
        output: str,
        reference: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.factual_correctness import FactualCorrectnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = FactualCorrectnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="factual_correctness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, reference=reference),
            cache_kwargs={"output": output, "reference": reference, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def noise_sensitivity(
        self,
        output: str,
        context: str,
        noisy_context: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.noise_sensitivity import NoiseSensitivityMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = NoiseSensitivityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="noise_sensitivity",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, context=context, noisy_context=noisy_context),
            cache_kwargs={"output": output, "context": context, "noisy_context": noisy_context, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def context_entity_recall(
        self,
        context: str,
        reference: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.context_entity_recall import ContextEntityRecallMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ContextEntityRecallMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="context_entity_recall",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(context=context, reference=reference),
            cache_kwargs={"context": context, "reference": reference, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=context,
        )

    def topic_adherence(
        self,
        output: str,
        allowed_topics: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.topic_adherence import TopicAdherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = TopicAdherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="topic_adherence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, allowed_topics=allowed_topics),
            cache_kwargs={"output": output, "allowed_topics": ",".join(allowed_topics), "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def response_completeness(
        self,
        output: str,
        query: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.response_completeness import ResponseCompletenessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ResponseCompletenessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="response_completeness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, query=query),
            cache_kwargs={"output": output, "query": query, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def sql_equivalence(
        self,
        output: str,
        reference: str,
        schema: str | None = None,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.sql_equivalence import SQLEquivalenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = SQLEquivalenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        cache_kw = {"output": output, "reference": reference, "threshold": str(t)}
        if schema:
            cache_kw["schema"] = schema
        return self._cached_judge_check(
            metric_name="sql_equivalence",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, reference=reference, schema=schema),
            cache_kwargs=cache_kw,
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def code_correctness(
        self,
        output: str,
        requirements: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.code_correctness import CodeCorrectnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CodeCorrectnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="code_correctness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, requirements=requirements),
            cache_kwargs={"output": output, "requirements": requirements, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def citation_accuracy(
        self,
        output: str,
        sources: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.citation_accuracy import CitationAccuracyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CitationAccuracyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="citation_accuracy",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, sources=sources),
            cache_kwargs={"output": output, "sources": "|".join(sources), "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def instruction_completeness(
        self,
        output: str,
        instructions: list[str],
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.instruction_completeness import InstructionCompletenessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = InstructionCompletenessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="instruction_completeness",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output=output, instructions=instructions),
            cache_kwargs={"output": output, "instructions": "|".join(instructions), "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output,
        )

    def comparative_quality(
        self,
        output_a: str,
        output_b: str,
        criteria: str,
        threshold: float | None = None,
        runs: int | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.comparative_quality import ComparativeQualityMetric
        t = threshold if threshold is not None else 0.5
        metric = ComparativeQualityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt
        return self._cached_judge_check(
            metric_name="comparative_quality",
            metric_factory=lambda: metric,
            coro_factory=lambda: metric.evaluate(output_a=output_a, output_b=output_b, criteria=criteria),
            cache_kwargs={"output_a": output_a, "output_b": output_b, "criteria": criteria, "threshold": str(t)},
            runs=runs,
            threshold=t,
            input_preview=output_a,
        )

    # --- Async LLM-as-judge checks ---

    async def ahallucination(
        self, output: str, context: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.hallucination import HallucinationMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = HallucinationMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("hallucination", model, output=output, context=context, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("hallucination")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, context=context)
        self._track_cost(result)
        self._cache.put(key, "hallucination", model, result)
        self.results.append(result)
        return result

    async def arelevance(
        self, output: str, query: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.relevance import RelevanceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RelevanceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("relevance", model, output=output, query=query, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("relevance")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, query=query)
        self._track_cost(result)
        self._cache.put(key, "relevance", model, result)
        self.results.append(result)
        return result

    async def atoxicity(
        self, output: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.toxicity import ToxicityMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = ToxicityMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("toxicity", model, output=output, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("toxicity")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output)
        self._track_cost(result)
        self._cache.put(key, "toxicity", model, result)
        self.results.append(result)
        return result

    async def arubric(
        self, output: str, criteria: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.rubric import RubricMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = RubricMetric(judge=self._get_judge())
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("rubric", model, output=output, criteria=criteria, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("rubric")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, criteria=criteria, threshold=t)
        self._track_cost(result)
        self._cache.put(key, "rubric", model, result)
        self.results.append(result)
        return result

    async def afluency(
        self, output: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.fluency import FluencyMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = FluencyMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("fluency", model, output=output, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("fluency")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output)
        self._track_cost(result)
        self._cache.put(key, "fluency", model, result)
        self.results.append(result)
        return result

    async def acoherence(
        self, output: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.coherence import CoherenceMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CoherenceMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("coherence", model, output=output, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("coherence")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output)
        self._track_cost(result)
        self._cache.put(key, "coherence", model, result)
        self.results.append(result)
        return result

    async def asentiment(
        self, output: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.sentiment import SentimentMetric
        t = threshold if threshold is not None else 0.5
        metric = SentimentMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("sentiment", model, output=output, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("sentiment")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output)
        self._track_cost(result)
        self._cache.put(key, "sentiment", model, result)
        self.results.append(result)
        return result

    async def acorrectness(
        self, output: str, expected: str, threshold: float | None = None,
        system_prompt: str | None = None,
    ) -> CheckResult:
        from checkllm.metrics.correctness import CorrectnessMetric
        t = threshold if threshold is not None else self.config.default_threshold
        metric = CorrectnessMetric(judge=self._get_judge(), threshold=t)
        if system_prompt is not None:
            metric.system_prompt = system_prompt

        model = getattr(self._get_judge(), "model", "unknown")
        key = _cache_key("correctness", model, output=output, expected=expected, threshold=str(t))
        cached = self._cache.get(key)
        if cached is not None:
            self.results.append(cached)
            return cached

        if not self._check_budget():
            result = self._make_budget_skip_result("correctness")
            self.results.append(result)
            return result

        async with self._semaphore:
            result = await metric.evaluate(output=output, expected=expected)
        self._track_cost(result)
        self._cache.put(key, "correctness", model, result)
        self.results.append(result)
        return result
