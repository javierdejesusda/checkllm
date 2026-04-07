"""YAML declarative eval configuration — load and run evaluations from YAML files.

Supports a promptfoo-inspired YAML format for defining providers, prompts,
test cases, and assertions declaratively.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment
from pydantic import BaseModel, Field

from checkllm.deterministic import DeterministicChecks
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.yaml_config")

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """A single LLM provider/model configuration."""

    id: str  # "openai", "anthropic", "gemini", etc.
    model: str = ""
    api_key: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PromptConfig(BaseModel):
    """A prompt template with optional default variables."""

    name: str
    template: str
    variables: dict[str, str] = Field(default_factory=dict)


class AssertionConfig(BaseModel):
    """A single assertion (check) to run on an output."""

    type: str  # "contains", "relevance", "max_tokens", etc.
    value: Any = None
    threshold: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class TestConfig(BaseModel):
    """A single test case with input, expected output, and assertions."""

    description: str = ""
    input: str = ""
    expected: str | None = None
    context: str | None = None
    query: str | None = None
    assertions: list[AssertionConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    """Top-level evaluation configuration loaded from YAML."""

    description: str = ""
    providers: list[ProviderConfig] = Field(default_factory=list)
    prompts: list[PromptConfig] = Field(default_factory=list)
    tests: list[TestConfig] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_jinja_env = Environment(
    loader=BaseLoader(),
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
)


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load an ``EvalConfig`` from a YAML file.

    Raises ``FileNotFoundError`` if *path* does not exist, and
    ``ValueError`` for invalid YAML content.
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

    return EvalConfig(**data)


def render_prompt(template: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2-style ``{{variable}}`` template with the given variables."""
    try:
        tpl = _jinja_env.from_string(template)
        return tpl.render(**variables)
    except Exception:
        # Fallback: simple string replacement for {{var}} patterns
        result = template
        for key, val in variables.items():
            result = result.replace("{{" + key + "}}", str(val))
            result = result.replace("{{ " + key + " }}", str(val))
        return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class YamlEvalRunner:
    """Run evaluations defined in a YAML config file.

    This runner iterates over every combination of provider x prompt x test,
    generates outputs via the provider, then runs all assertions on each output.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self._deterministic = DeterministicChecks()

    # -- public API ----------------------------------------------------------

    async def run(self) -> list[dict[str, Any]]:
        """Run all tests against all providers with all prompts.

        Returns a list of result dicts, one per (provider, prompt, test)
        combination, each containing:

        - ``provider``: provider id
        - ``model``: model name
        - ``prompt``: prompt name
        - ``test``: test description
        - ``output``: generated text
        - ``results``: list of ``CheckResult`` objects
        - ``passed``: overall pass/fail
        - ``duration_ms``: wall-clock milliseconds for this combination
        """
        all_results: list[dict[str, Any]] = []
        concurrency = int(self.config.settings.get("max_concurrency", 5))
        semaphore = asyncio.Semaphore(concurrency)

        providers = self.config.providers
        prompts = self.config.prompts
        tests = self.config.tests

        # If no prompts defined, create a passthrough prompt using the test input
        if not prompts:
            prompts = [PromptConfig(name="default", template="{{input}}")]

        for provider_cfg in providers:
            judge = self._create_judge(provider_cfg)

            for prompt_cfg in prompts:
                for test_cfg in tests:
                    async with semaphore:
                        t0 = time.perf_counter()

                        # Build variables for prompt rendering
                        variables: dict[str, Any] = {
                            "input": test_cfg.input,
                            **prompt_cfg.variables,
                            **test_cfg.metadata,
                        }
                        if test_cfg.context:
                            variables["context"] = test_cfg.context
                        if test_cfg.query:
                            variables["query"] = test_cfg.query
                        if test_cfg.expected:
                            variables["expected"] = test_cfg.expected

                        rendered = render_prompt(prompt_cfg.template, variables)

                        # Generate output from the provider
                        output = await self._generate(rendered, provider_cfg)

                        # Run assertions
                        check_results = await self._run_test(
                            test_cfg, output, judge
                        )

                        elapsed = int((time.perf_counter() - t0) * 1000)
                        passed = all(r.passed for r in check_results)

                        all_results.append(
                            {
                                "provider": provider_cfg.id,
                                "model": provider_cfg.model,
                                "prompt": prompt_cfg.name,
                                "test": test_cfg.description,
                                "input": test_cfg.input,
                                "output": output,
                                "results": check_results,
                                "passed": passed,
                                "duration_ms": elapsed,
                            }
                        )

                        status = "PASS" if passed else "FAIL"
                        logger.info(
                            "[%s] %s / %s / %s (%d assertions, %dms)",
                            status,
                            provider_cfg.id,
                            prompt_cfg.name,
                            test_cfg.description,
                            len(check_results),
                            elapsed,
                        )

        return all_results

    # -- internal helpers ----------------------------------------------------

    def _create_judge(self, provider: ProviderConfig):
        """Create a judge backend from a provider config.

        Uses ``checkllm.providers.create_judge`` for known backends and falls
        back to the default OpenAI judge when the provider id is not
        recognised as a judge backend.
        """
        from checkllm.providers import create_judge

        kwargs: dict[str, Any] = {}
        if provider.model:
            kwargs["model"] = provider.model
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        kwargs.update(provider.options)

        try:
            return create_judge(provider.id, **kwargs)
        except (ValueError, Exception) as exc:
            logger.warning(
                "Could not create judge for provider '%s': %s. "
                "Falling back to OpenAI judge.",
                provider.id,
                exc,
            )
            from checkllm.judge import OpenAIJudge

            return OpenAIJudge(model=provider.model or "gpt-4o")

    async def _generate(self, prompt: str, provider: ProviderConfig) -> str:
        """Generate LLM output using the given provider."""
        pid = provider.id.lower()
        model = provider.model

        try:
            if pid == "anthropic":
                try:
                    from anthropic import AsyncAnthropic
                except ImportError:
                    return "[anthropic package not installed]"
                client = AsyncAnthropic(
                    api_key=provider.api_key,
                    **(provider.options.get("client", {})),
                )
                response = await client.messages.create(
                    model=model or "claude-sonnet-4-6",
                    max_tokens=provider.options.get("max_tokens", 1024),
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text if response.content else ""

            elif pid in ("gemini", "google"):
                try:
                    import google.generativeai as genai
                except ImportError:
                    return "[google-generativeai package not installed]"
                genai.configure(api_key=provider.api_key)
                gen_model = genai.GenerativeModel(model or "gemini-pro")
                response = await asyncio.to_thread(
                    gen_model.generate_content, prompt
                )
                return response.text

            else:
                # Default: OpenAI-compatible
                from openai import AsyncOpenAI

                client = AsyncOpenAI(
                    api_key=provider.api_key,
                    **(provider.options.get("client", {})),
                )
                response = await client.chat.completions.create(
                    model=model or "gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=provider.options.get("temperature", 0.0),
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Generation failed for %s/%s: %s", pid, model, exc)
            return f"[Generation failed: {exc}]"

    async def _run_test(
        self,
        test: TestConfig,
        prompt_output: str,
        judge: Any,
    ) -> list[CheckResult]:
        """Run all assertions for a single test case."""
        results: list[CheckResult] = []
        for assertion in test.assertions:
            try:
                result = await self._run_assertion(
                    assertion, prompt_output, test, judge
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Assertion '%s' raised: %s", assertion.type, exc
                )
                results.append(
                    CheckResult(
                        passed=False,
                        score=0.0,
                        reasoning=f"Assertion error: {exc}",
                        cost=0.0,
                        latency_ms=0,
                        metric_name=assertion.type,
                    )
                )
        return results

    async def _run_assertion(
        self,
        assertion: AssertionConfig,
        output: str,
        test: TestConfig,
        judge: Any,
    ) -> CheckResult:
        """Map an assertion type to the appropriate check and execute it."""
        atype = assertion.type.lower().strip()
        value = assertion.value
        threshold = assertion.threshold

        # -- Deterministic checks -------------------------------------------

        if atype == "contains":
            return self._deterministic.contains(output, str(value))

        if atype == "not_contains":
            return self._deterministic.not_contains(output, str(value))

        if atype == "exact_match":
            ignore_case = assertion.options.get("ignore_case", False)
            return self._deterministic.exact_match(
                output, str(value), ignore_case=ignore_case
            )

        if atype == "regex":
            return self._deterministic.regex(output, str(value))

        if atype == "starts_with":
            return self._deterministic.starts_with(output, str(value))

        if atype == "ends_with":
            return self._deterministic.ends_with(output, str(value))

        if atype == "max_tokens":
            return self._deterministic.max_tokens(output, int(value))

        if atype == "min_tokens":
            return self._deterministic.min_tokens(output, int(value))

        if atype == "word_count":
            min_w = assertion.options.get("min")
            max_w = assertion.options.get("max")
            if value is not None and min_w is None and max_w is None:
                # Treat bare value as max_words
                max_w = int(value)
            return self._deterministic.word_count(
                output,
                min_words=int(min_w) if min_w is not None else None,
                max_words=int(max_w) if max_w is not None else None,
            )

        if atype == "is_json":
            return self._deterministic.is_json(output)

        if atype == "json_schema":
            # value should be a dict representing a Pydantic model schema,
            # but for simplicity we just validate JSON parsability here
            return self._deterministic.is_json(output)

        if atype == "no_pii":
            patterns = assertion.options.get("patterns")
            return self._deterministic.no_pii(output, patterns=patterns)

        if atype == "bleu":
            ref = str(value) if value else (test.expected or "")
            t = threshold if threshold is not None else 0.5
            return self._deterministic.bleu(output, ref, threshold=t)

        if atype == "rouge_l":
            ref = str(value) if value else (test.expected or "")
            t = threshold if threshold is not None else 0.5
            return self._deterministic.rouge_l(output, ref, threshold=t)

        if atype == "similarity":
            expected = str(value) if value else (test.expected or "")
            t = threshold if threshold is not None else 0.8
            return self._deterministic.similarity(output, expected, threshold=t)

        if atype == "is_valid_python":
            return self._deterministic.is_valid_python(output)

        if atype == "language":
            return self._deterministic.language(output, str(value))

        # -- LLM-as-judge checks (async) ------------------------------------

        t = threshold if threshold is not None else 0.8

        if atype == "relevance":
            from checkllm.metrics.relevance import RelevanceMetric

            query = test.query or test.input
            metric = RelevanceMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, query=query)

        if atype == "hallucination":
            from checkllm.metrics.hallucination import HallucinationMetric

            context = test.context or test.input
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

            expected = str(value) if value else (test.expected or test.input)
            metric = CorrectnessMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, expected=expected)

        if atype == "faithfulness":
            from checkllm.metrics.faithfulness import FaithfulnessMetric

            context = test.context or test.input
            metric = FaithfulnessMetric(judge=judge, threshold=t)
            return await metric.evaluate(
                output=output, context=context, query=test.query
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

            categories = assertion.options.get("categories")
            metric = BiasMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, categories=categories)

        if atype == "summarization":
            from checkllm.metrics.summarization import SummarizationMetric

            source = str(value) if value else (test.context or test.input)
            metric = SummarizationMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, source=source)

        if atype == "instruction_following":
            from checkllm.metrics.instruction_following import (
                InstructionFollowingMetric,
            )

            instructions = str(value) if value else test.input
            metric = InstructionFollowingMetric(judge=judge, threshold=t)
            return await metric.evaluate(
                output=output, instructions=instructions
            )

        if atype == "answer_completeness":
            from checkllm.metrics.answer_completeness import (
                AnswerCompletenessMetric,
            )

            query = test.query or test.input
            metric = AnswerCompletenessMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, query=query)

        if atype == "context_relevance":
            from checkllm.metrics.context_relevance import (
                ContextRelevanceMetric,
            )

            context = test.context or test.input
            query = test.query or test.input
            metric = ContextRelevanceMetric(judge=judge, threshold=t)
            return await metric.evaluate(context=context, query=query)

        if atype == "groundedness":
            from checkllm.metrics.groundedness import GroundednessMetric

            sources = [test.context or test.input]
            if isinstance(value, list):
                sources = [str(s) for s in value]
            elif value:
                sources = [str(value)]
            metric = GroundednessMetric(judge=judge, threshold=t)
            return await metric.evaluate(output=output, sources=sources)

        if atype == "consistency":
            from checkllm.metrics.consistency import ConsistencyMetric

            # consistency expects multiple outputs; use value as a list
            outputs = [output]
            if isinstance(value, list):
                outputs.extend(str(v) for v in value)
            metric = ConsistencyMetric(judge=judge, threshold=t)
            return await metric.evaluate(outputs=outputs)

        # -- Unknown assertion type -----------------------------------------

        return CheckResult(
            passed=False,
            score=0.0,
            reasoning=f"Unknown assertion type: '{atype}'",
            cost=0.0,
            latency_ms=0,
            metric_name=atype,
        )
