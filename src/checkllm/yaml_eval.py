"""YAML-based evaluation system for non-programmatic LLM evaluation.

Provides a promptfoo-style YAML configuration format that supports
prompt templates with variable substitution, multiple providers,
and a rich set of assertion types including both deterministic
checks and LLM-as-judge metrics.

Usage::

    from checkllm.yaml_eval import YAMLEvaluator

    evaluator = YAMLEvaluator()
    results = await evaluator.run("checkllm.yaml")
    print(results.summary())

Example YAML config::

    description: "Customer support chatbot evaluation"
    judge:
      backend: openai
      model: gpt-4o

    prompts:
      - "You are a helpful customer support agent. Answer: {{query}}"
      - "As a support rep, help with: {{query}}"

    providers:
      - openai:gpt-4o
      - anthropic:claude-sonnet-4-6

    tests:
      - vars:
          query: "How do I return an item?"
        assert:
          - type: contains
            value: "return policy"
          - type: relevance
            threshold: 0.8
          - type: no_pii
          - type: max_tokens
            value: 500

    settings:
      budget: 5.0
      threshold: 0.8
      parallel: true
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from checkllm.deterministic import DeterministicChecks
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.yaml_eval")


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge used in evaluations."""

    backend: str = "auto"
    model: str = ""


class AssertionConfig(BaseModel):
    """A single assertion to run on an LLM output.

    Attributes:
        type: The assertion type, e.g. "contains", "relevance", "no_pii".
        value: Optional value argument for the assertion.
        threshold: Optional score threshold for metric-based assertions.
    """

    type: str
    value: Any = None
    threshold: float | None = None

    class Config:
        populate_by_name = True


class EvalTestConfig(BaseModel):
    """A single test case with variables and assertions.

    Attributes:
        vars: Template variables for prompt rendering.
        assert_: List of assertions to run on the output.
        description: Optional human-readable test description.
    """

    vars: dict[str, str] = Field(default_factory=dict)
    assert_: list[AssertionConfig] = Field(default_factory=list, alias="assert")
    description: str = ""

    class Config:
        populate_by_name = True


class EvalSettings(BaseModel):
    """Settings controlling evaluation behavior.

    Attributes:
        budget: Maximum USD to spend on judge calls.
        threshold: Default pass/fail threshold for metric assertions.
        parallel: Whether to run tests in parallel.
        cache: Whether to cache judge responses.
    """

    budget: float = 10.0
    threshold: float = 0.8
    parallel: bool = True
    cache: bool = True


class YAMLEvalConfig(BaseModel):
    """Parsed YAML evaluation configuration.

    Attributes:
        description: Human-readable description of the evaluation.
        judge: Configuration for the LLM judge.
        prompts: List of prompt templates with ``{{var}}`` placeholders.
        providers: List of provider strings like ``"openai:gpt-4o"``.
        tests: List of test cases to evaluate.
        settings: Evaluation settings.
    """

    description: str = ""
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    prompts: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    tests: list[EvalTestConfig] = Field(default_factory=list)
    settings: EvalSettings = Field(default_factory=EvalSettings)


class TestResult(BaseModel):
    """Result for a single test case execution.

    Attributes:
        test_index: Index of the test in the config file.
        prompt: The rendered prompt that was evaluated.
        provider: The provider used for generation.
        vars: The template variables used.
        assertions: Per-assertion results.
        passed: Whether all assertions passed.
        output: The LLM-generated output.
        duration_ms: Wall-clock time in milliseconds.
    """

    test_index: int = 0
    prompt: str = ""
    provider: str = ""
    vars: dict[str, str] = Field(default_factory=dict)
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = True
    output: str = ""
    duration_ms: float = 0.0


class YAMLEvalResult(BaseModel):
    """Results from a YAML-based evaluation run.

    Attributes:
        config_path: Path to the YAML config file.
        description: Description from the config.
        total_tests: Total number of test assertions run.
        passed: Number of assertions that passed.
        failed: Number of assertions that failed.
        results: Per-test-case results.
        cost: Total cost in USD.
        duration_ms: Total wall-clock time in milliseconds.
    """

    config_path: str = ""
    description: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: list[TestResult] = Field(default_factory=list)
    cost: float = 0.0
    duration_ms: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation results.

        Returns:
            A multi-line string with pass/fail counts, cost, and
            per-test status.
        """
        pct = (self.passed / self.total_tests * 100) if self.total_tests > 0 else 0.0
        lines = [
            f"YAML Evaluation: {self.description or self.config_path}",
            f"Results: {self.passed}/{self.total_tests} passed ({pct:.0f}%)",
            f"Failed: {self.failed}",
            f"Cost: ${self.cost:.4f}",
            f"Duration: {self.duration_ms:.0f}ms",
            "",
        ]
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            provider_label = f" [{result.provider}]" if result.provider else ""
            lines.append(f"  [{status}]{provider_label} Test {result.test_index + 1}")
            for a in result.assertions:
                a_status = "PASS" if a.get("passed", False) else "FAIL"
                lines.append(f"    [{a_status}] {a.get('type', '?')}: {a.get('reasoning', '')[:80]}")
        return "\n".join(lines)


def _render_template(template: str, variables: dict[str, str]) -> str:
    """Render a ``{{variable}}`` template with simple substitution.

    Supports both ``{{var}}`` and ``{{ var }}`` syntax.

    Args:
        template: The template string with ``{{var}}`` placeholders.
        variables: A mapping of variable names to values.

    Returns:
        The rendered string with all known variables replaced.
    """
    result = template
    for key, val in variables.items():
        result = result.replace("{{" + key + "}}", str(val))
        result = result.replace("{{ " + key + " }}", str(val))
    return result


def load_yaml_eval_config(path: str | Path) -> YAMLEvalConfig:
    """Load a YAML evaluation configuration from a file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A parsed YAMLEvalConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level in {p}")

    config_data: dict[str, Any] = {}

    config_data["description"] = data.get("description", "")

    judge_data = data.get("judge")
    if isinstance(judge_data, dict):
        config_data["judge"] = JudgeConfig(**judge_data)

    config_data["prompts"] = data.get("prompts", [])
    config_data["providers"] = data.get("providers", [])

    tests_raw = data.get("tests", [])
    tests: list[EvalTestConfig] = []
    for t in tests_raw:
        if not isinstance(t, dict):
            continue
        test_kwargs: dict[str, Any] = {}
        if "vars" in t:
            test_kwargs["vars"] = t["vars"]
        if "description" in t:
            test_kwargs["description"] = t["description"]
        assertions_raw = t.get("assert", [])
        assertions: list[AssertionConfig] = []
        for a in assertions_raw:
            if isinstance(a, dict):
                assertions.append(AssertionConfig(**a))
        test_kwargs["assert_"] = assertions
        tests.append(EvalTestConfig(**test_kwargs))
    config_data["tests"] = tests

    settings_data = data.get("settings")
    if isinstance(settings_data, dict):
        config_data["settings"] = EvalSettings(**settings_data)

    return YAMLEvalConfig(**config_data)


class YAMLEvaluator:
    """Run evaluations from YAML configuration files.

    This evaluator parses a YAML config file and runs all defined test
    cases against all prompts and providers, executing both deterministic
    and LLM-as-judge assertions.

    Usage::

        from checkllm.yaml_eval import YAMLEvaluator

        evaluator = YAMLEvaluator()
        results = await evaluator.run("checkllm.yaml")
        print(results.summary())
    """

    def __init__(self) -> None:
        self._deterministic = DeterministicChecks()

    def load_config(self, path: str | Path) -> YAMLEvalConfig:
        """Parse and validate a YAML config file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A parsed YAMLEvalConfig object.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML is invalid or malformed.
        """
        return load_yaml_eval_config(path)

    async def run(self, config_path: str | Path) -> YAMLEvalResult:
        """Execute all tests defined in the YAML config.

        Args:
            config_path: Path to the YAML evaluation config file.

        Returns:
            A YAMLEvalResult with aggregate and per-test results.
        """
        t0 = time.perf_counter()
        config = self.load_config(config_path)
        judge = self._create_judge(config.judge)

        prompts = config.prompts
        if not prompts:
            prompts = ["{{query}}"]

        providers = config.providers
        if not providers:
            providers = ["default"]

        all_results: list[TestResult] = []
        total_passed = 0
        total_failed = 0
        total_cost = 0.0

        budget = config.settings.budget
        budget_exceeded = False

        for test_idx, test_cfg in enumerate(config.tests):
            if budget_exceeded:
                break
            for prompt_template in prompts:
                if budget_exceeded:
                    break
                for provider_str in providers:
                    if budget_exceeded:
                        break
                    rendered = _render_template(prompt_template, test_cfg.vars)

                    output = await self._generate_output(
                        rendered, provider_str, config.judge
                    )

                    assertion_results: list[dict[str, Any]] = []
                    test_passed = True

                    for assertion in test_cfg.assert_:
                        if budget > 0 and total_cost >= budget:
                            budget_exceeded = True
                            break
                        check_result = await self._resolve_assertion(
                            assertion, output, test_cfg, judge, config.settings,
                        )
                        a_passed = check_result.passed
                        assertion_results.append({
                            "type": assertion.type,
                            "passed": a_passed,
                            "score": check_result.score,
                            "reasoning": check_result.reasoning or "",
                            "cost": check_result.cost,
                        })
                        total_cost += check_result.cost
                        if a_passed:
                            total_passed += 1
                        else:
                            total_failed += 1
                            test_passed = False

                    all_results.append(TestResult(
                        test_index=test_idx,
                        prompt=rendered,
                        provider=provider_str,
                        vars=test_cfg.vars,
                        assertions=assertion_results,
                        passed=test_passed,
                        output=output,
                    ))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return YAMLEvalResult(
            config_path=str(config_path),
            description=config.description,
            total_tests=total_passed + total_failed,
            passed=total_passed,
            failed=total_failed,
            results=all_results,
            cost=total_cost,
            duration_ms=elapsed_ms,
        )

    async def run_from_config(self, config: YAMLEvalConfig) -> YAMLEvalResult:
        """Execute all tests from an already-parsed config object.

        Args:
            config: A parsed YAMLEvalConfig object.

        Returns:
            A YAMLEvalResult with aggregate and per-test results.
        """
        t0 = time.perf_counter()
        judge = self._create_judge(config.judge)

        prompts = config.prompts or ["{{query}}"]
        providers = config.providers or ["default"]

        all_results: list[TestResult] = []
        total_passed = 0
        total_failed = 0
        total_cost = 0.0

        budget = config.settings.budget
        budget_exceeded = False

        for test_idx, test_cfg in enumerate(config.tests):
            if budget_exceeded:
                break
            for prompt_template in prompts:
                if budget_exceeded:
                    break
                for provider_str in providers:
                    if budget_exceeded:
                        break
                    rendered = _render_template(prompt_template, test_cfg.vars)

                    output = await self._generate_output(
                        rendered, provider_str, config.judge
                    )

                    assertion_results: list[dict[str, Any]] = []
                    test_passed = True

                    for assertion in test_cfg.assert_:
                        if budget > 0 and total_cost >= budget:
                            budget_exceeded = True
                            break
                        check_result = await self._resolve_assertion(
                            assertion, output, test_cfg, judge, config.settings,
                        )
                        a_passed = check_result.passed
                        assertion_results.append({
                            "type": assertion.type,
                            "passed": a_passed,
                            "score": check_result.score,
                            "reasoning": check_result.reasoning or "",
                            "cost": check_result.cost,
                        })
                        total_cost += check_result.cost
                        if a_passed:
                            total_passed += 1
                        else:
                            total_failed += 1
                            test_passed = False

                    all_results.append(TestResult(
                        test_index=test_idx,
                        prompt=rendered,
                        provider=provider_str,
                        vars=test_cfg.vars,
                        assertions=assertion_results,
                        passed=test_passed,
                        output=output,
                    ))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return YAMLEvalResult(
            config_path="<in-memory>",
            description=config.description,
            total_tests=total_passed + total_failed,
            passed=total_passed,
            failed=total_failed,
            results=all_results,
            cost=total_cost,
            duration_ms=elapsed_ms,
        )

    def _create_judge(self, judge_config: JudgeConfig) -> Any:
        """Create a judge backend from the config.

        Args:
            judge_config: The judge configuration.

        Returns:
            A judge backend instance, or None if unavailable.
        """
        try:
            from checkllm.providers import create_judge

            backend = judge_config.backend
            if backend == "auto":
                backend = "openai"

            kwargs: dict[str, Any] = {}
            if judge_config.model:
                kwargs["model"] = judge_config.model

            return create_judge(backend, **kwargs)
        except Exception as exc:
            logger.warning("Could not create judge: %s", exc)
            return None

    async def _generate_output(
        self,
        prompt: str,
        provider_str: str,
        judge_config: JudgeConfig,
    ) -> str:
        """Generate LLM output from the given provider.

        Args:
            prompt: The rendered prompt text.
            provider_str: Provider string like ``"openai:gpt-4o"`` or
                ``"default"``.
            judge_config: The judge configuration for fallback model info.

        Returns:
            The generated text, or an error placeholder on failure.
        """
        parts = provider_str.split(":", 1)
        provider_id = parts[0].lower()
        model = parts[1] if len(parts) > 1 else ""

        if provider_id == "default":
            provider_id = judge_config.backend
            if provider_id == "auto":
                provider_id = "openai"
            if not model:
                model = judge_config.model or "gpt-4o"

        try:
            if provider_id == "anthropic":
                try:
                    from anthropic import AsyncAnthropic
                except ImportError:
                    return "[anthropic package not installed]"
                client = AsyncAnthropic()
                response = await client.messages.create(
                    model=model or "claude-sonnet-4-6",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text if response.content else ""
            else:
                from openai import AsyncOpenAI
                client = AsyncOpenAI()
                response = await client.chat.completions.create(
                    model=model or "gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Generation failed for %s/%s: %s", provider_id, model, exc)
            return f"[Generation failed: {exc}]"

    async def _resolve_assertion(
        self,
        assertion: AssertionConfig,
        output: str,
        test: EvalTestConfig,
        judge: Any,
        settings: EvalSettings,
    ) -> CheckResult:
        """Map an assertion type string to the appropriate check and execute it.

        This method handles both deterministic checks (contains, no_pii,
        max_tokens, etc.) and LLM-as-judge metrics (relevance, toxicity,
        hallucination, etc.).

        Args:
            assertion: The assertion configuration.
            output: The LLM output to evaluate.
            test: The test case context.
            judge: The judge backend for LLM-based assertions.
            settings: Evaluation settings including default threshold.

        Returns:
            A CheckResult with pass/fail status and score.
        """
        atype = assertion.type.lower().strip()
        value = assertion.value
        threshold = assertion.threshold

        if atype == "contains":
            return self._deterministic.contains(output, str(value or ""))

        if atype == "not_contains":
            return self._deterministic.not_contains(output, str(value or ""))

        if atype == "exact_match":
            return self._deterministic.exact_match(output, str(value or ""))

        if atype == "regex":
            return self._deterministic.regex(output, str(value or ""))

        if atype == "starts_with":
            return self._deterministic.starts_with(output, str(value or ""))

        if atype == "ends_with":
            return self._deterministic.ends_with(output, str(value or ""))

        if atype == "max_tokens":
            return self._deterministic.max_tokens(output, int(value or 500))

        if atype == "min_tokens":
            return self._deterministic.min_tokens(output, int(value or 1))

        if atype == "is_json":
            return self._deterministic.is_json(output)

        if atype == "no_pii":
            return self._deterministic.no_pii(output)

        if atype == "word_count":
            max_w = int(value) if value is not None else None
            return self._deterministic.word_count(
                output, max_words=max_w
            )

        if atype == "bleu":
            ref = str(value) if value else ""
            t = threshold if threshold is not None else 0.5
            return self._deterministic.bleu(output, ref, threshold=t)

        if atype == "rouge_l":
            ref = str(value) if value else ""
            t = threshold if threshold is not None else 0.5
            return self._deterministic.rouge_l(output, ref, threshold=t)

        if atype == "similarity":
            expected = str(value) if value else ""
            t = threshold if threshold is not None else 0.8
            return self._deterministic.similarity(output, expected, threshold=t)

        if atype == "is_valid_python":
            return self._deterministic.is_valid_python(output)

        if atype == "language":
            return self._deterministic.language(output, str(value or "en"))

        if judge is None:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"LLM judge required for assertion type '{atype}' but none configured",
                cost=0.0,
                latency_ms=0,
                metric_name=atype,
            )

        t = threshold if threshold is not None else settings.threshold

        if atype == "relevance":
            from checkllm.metrics.relevance import RelevanceMetric
            query = test.vars.get("query", test.vars.get("input", ""))
            metric = RelevanceMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, query=query)

        if atype == "hallucination":
            from checkllm.metrics.hallucination import HallucinationMetric
            context = test.vars.get("context", test.vars.get("input", ""))
            metric = HallucinationMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, context=context)

        if atype == "toxicity":
            from checkllm.metrics.toxicity import ToxicityMetric
            metric = ToxicityMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output)

        if atype == "fluency":
            from checkllm.metrics.fluency import FluencyMetric
            metric = FluencyMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output)

        if atype == "coherence":
            from checkllm.metrics.coherence import CoherenceMetric
            metric = CoherenceMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output)

        if atype == "correctness":
            from checkllm.metrics.correctness import CorrectnessMetric
            expected = str(value) if value else test.vars.get("expected", "")
            metric = CorrectnessMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, expected=expected)

        if atype == "faithfulness":
            from checkllm.metrics.faithfulness import FaithfulnessMetric
            context = test.vars.get("context", test.vars.get("input", ""))
            query = test.vars.get("query", "")
            metric = FaithfulnessMetric(judge=judge, threshold=t)
            return await metric.evaluate(
                output=output, context=context, query=query
            )

        if atype == "rubric":
            from checkllm.metrics.rubric import RubricMetric
            criteria = str(value) if value else "Output should be accurate, helpful, and well-structured."
            metric = RubricMetric(judge=judge)
            return await metric.evaluate(
                output=output, criteria=criteria, threshold=t
            )

        if atype == "sentiment":
            from checkllm.metrics.sentiment import SentimentMetric
            metric = SentimentMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output)

        if atype == "bias":
            from checkllm.metrics.bias import BiasMetric
            metric = BiasMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output)

        if atype == "summarization":
            from checkllm.metrics.summarization import SummarizationMetric
            source = str(value) if value else test.vars.get("context", "")
            metric = SummarizationMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, source=source)

        if atype == "instruction_following":
            from checkllm.metrics.instruction_following import InstructionFollowingMetric
            instructions = str(value) if value else test.vars.get("input", "")
            metric = InstructionFollowingMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, instructions=instructions)

        if atype == "role_adherence":
            from checkllm.metrics.role_adherence import RoleAdherenceMetric
            role = str(value) if value else test.vars.get("role", "assistant")
            metric = RoleAdherenceMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, role=role)

        if atype == "groundedness":
            from checkllm.metrics.groundedness import GroundednessMetric
            sources = [str(value)] if value else [test.vars.get("context", "")]
            metric = GroundednessMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, sources=sources)

        return CheckResult(
            passed=False,
            score=0.0,
            reasoning=f"Unknown assertion type: '{atype}'",
            cost=0.0,
            latency_ms=0,
            metric_name=atype,
        )
