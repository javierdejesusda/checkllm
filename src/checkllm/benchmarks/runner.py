"""BenchmarkRunner and related models for evaluating LLMs on benchmark datasets."""
from __future__ import annotations

import asyncio
import re
import time

from pydantic import BaseModel, Field

from checkllm.benchmarks.datasets import load_benchmark
from checkllm.judge import JudgeBackend


_MC_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

_OPEN_SYSTEM_PROMPT = (
    "Answer the question as accurately and concisely as possible."
)

_MATH_SYSTEM_PROMPT = (
    "Solve the math problem. Provide only the final numeric answer, "
    "with no units or explanation."
)


class BenchmarkResult(BaseModel):
    """Result of running a benchmark against an LLM.

    Attributes:
        benchmark: The benchmark name.
        total: Total number of samples evaluated.
        correct: Number of correct answers.
        accuracy: Fraction of correct answers (0.0–1.0).
        by_category: Per-category accuracy mapping.
        cost: Estimated total USD cost of evaluation.
        latency_ms: Total wall-clock time in milliseconds.
    """

    benchmark: str
    total: int
    correct: int
    accuracy: float = Field(ge=0.0, le=1.0)
    by_category: dict[str, float] = Field(default_factory=dict)
    cost: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)

    def summary(self) -> str:
        """Return a human-readable summary of the benchmark result.

        Returns:
            A multi-line string summarising accuracy, cost, and per-category scores.
        """
        lines = [
            f"Benchmark : {self.benchmark}",
            f"Accuracy  : {self.accuracy:.1%}  ({self.correct}/{self.total})",
            f"Cost      : ${self.cost:.4f}",
            f"Latency   : {self.latency_ms} ms",
        ]
        if self.by_category:
            lines.append("By category:")
            for cat, acc in sorted(self.by_category.items()):
                lines.append(f"  {cat}: {acc:.1%}")
        return "\n".join(lines)


class BenchmarkSuite(BaseModel):
    """Configuration for running multiple benchmarks.

    Attributes:
        benchmarks: List of benchmark names to run.
        limit_per_benchmark: Maximum samples to evaluate per benchmark.
    """

    benchmarks: list[str]
    limit_per_benchmark: int = 100


class BenchmarkRunner:
    """Evaluates an LLM backend against built-in benchmark datasets.

    Args:
        provider: A JudgeBackend implementation used to query the model.
    """

    def __init__(self, provider: JudgeBackend) -> None:
        self._provider = provider

    def run(self, benchmark: str, limit: int | None = None) -> BenchmarkResult:
        """Run a benchmark synchronously.

        Args:
            benchmark: The benchmark name to run.
            limit: Maximum number of samples to evaluate.

        Returns:
            A BenchmarkResult with accuracy and per-category breakdown.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.arun(benchmark, limit)
        )

    async def arun(self, benchmark: str, limit: int | None = None) -> BenchmarkResult:
        """Run a benchmark asynchronously.

        Args:
            benchmark: The benchmark name to run.
            limit: Maximum number of samples to evaluate.

        Returns:
            A BenchmarkResult with accuracy and per-category breakdown.
        """
        dataset = load_benchmark(benchmark, limit=limit)
        total = len(dataset.samples)
        correct = 0
        total_cost = 0.0
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}

        start = time.monotonic()

        for sample in dataset.samples:
            prompt, system_prompt = self._build_prompt(benchmark, sample)
            response = await self._provider.evaluate(
                prompt=prompt, system_prompt=system_prompt
            )
            total_cost += response.cost

            answer_text = response.raw_output or response.reasoning
            is_correct = self._check_answer(
                benchmark=benchmark,
                model_text=answer_text,
                correct_answer=sample.correct_answer,
            )
            if is_correct:
                correct += 1

            cat = sample.category or "uncategorized"
            category_total[cat] = category_total.get(cat, 0) + 1
            category_correct[cat] = category_correct.get(cat, 0) + (1 if is_correct else 0)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        by_category = {
            cat: category_correct.get(cat, 0) / category_total[cat]
            for cat in category_total
        }
        accuracy = correct / total if total > 0 else 0.0

        return BenchmarkResult(
            benchmark=benchmark,
            total=total,
            correct=correct,
            accuracy=accuracy,
            by_category=by_category,
            cost=total_cost,
            latency_ms=elapsed_ms,
        )

    def _build_prompt(
        self, benchmark: str, sample
    ) -> tuple[str, str]:
        """Construct the prompt and system prompt for a sample.

        Args:
            benchmark: The benchmark name.
            sample: A BenchmarkSample instance.

        Returns:
            A tuple of (prompt, system_prompt).
        """
        if benchmark == "gsm8k":
            system = _MATH_SYSTEM_PROMPT
            prompt = sample.question
        elif sample.choices:
            system = _MC_SYSTEM_PROMPT
            choices_text = "\n".join(sample.choices)
            prompt = f"{sample.question}\n\n{choices_text}"
        else:
            system = _OPEN_SYSTEM_PROMPT
            prompt = sample.question
        return prompt, system

    def _check_answer(
        self, benchmark: str, model_text: str, correct_answer: str
    ) -> bool:
        """Determine whether the model's response matches the correct answer.

        Args:
            benchmark: The benchmark name, used to choose comparison strategy.
            model_text: The raw text produced by the model.
            correct_answer: The ground-truth answer.

        Returns:
            True if the model's answer matches the correct answer.
        """
        model_text = (model_text or "").strip()
        if benchmark == "mmlu":
            return self._check_multiple_choice(model_text, correct_answer)
        if benchmark == "gsm8k":
            return self._check_numeric(model_text, correct_answer)
        return self._check_text(model_text, correct_answer)

    @staticmethod
    def _check_multiple_choice(model_text: str, correct_answer: str) -> bool:
        """Extract and compare the chosen letter from a multiple-choice response.

        Args:
            model_text: The model's response text.
            correct_answer: The expected letter (A–D).

        Returns:
            True if the extracted letter matches the correct answer.
        """
        match = re.search(r"\b([A-D])\b", model_text, re.IGNORECASE)
        if match:
            return match.group(1).upper() == correct_answer.upper()
        if model_text:
            return model_text[0].upper() == correct_answer.upper()
        return False

    @staticmethod
    def _check_numeric(model_text: str, correct_answer: str) -> bool:
        """Compare numeric values extracted from the model's response.

        Args:
            model_text: The model's response text.
            correct_answer: The expected numeric value as a string.

        Returns:
            True if the extracted number equals the correct answer.
        """
        numbers = re.findall(r"-?\d+(?:\.\d+)?", model_text)
        if not numbers:
            return False
        try:
            expected = float(correct_answer.replace(",", ""))
            actual = float(numbers[-1].replace(",", ""))
            return abs(actual - expected) < 1e-6
        except ValueError:
            return False

    @staticmethod
    def _check_text(model_text: str, correct_answer: str) -> bool:
        """Compare open-ended responses via case-insensitive containment.

        Args:
            model_text: The model's response text.
            correct_answer: The expected answer text.

        Returns:
            True if either text contains the other (case-insensitive).
        """
        model_lower = model_text.lower()
        answer_lower = correct_answer.lower()
        return answer_lower in model_lower or model_lower in answer_lower
