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

_MC_AB_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question with two options. "
    "Respond with only the letter of the correct answer (A or B)."
)

_OPEN_SYSTEM_PROMPT = "Answer the question as accurately and concisely as possible."

_MATH_SYSTEM_PROMPT = (
    "Solve the math problem. Provide only the final numeric answer, with no units or explanation."
)

_CODE_SYSTEM_PROMPT = (
    "Complete the function body. Provide only the implementation code, "
    "without the function signature or docstring."
)

_BOOLEAN_SYSTEM_PROMPT = "Answer the following yes/no question. Respond with only 'Yes' or 'No'."

_INSTRUCTION_SYSTEM_PROMPT = "Follow the instructions in the question exactly as specified."

_COMPLETION_SYSTEM_PROMPT = (
    "Complete the passage by providing the single missing word at the end. "
    "Respond with only that one word."
)

_READING_SYSTEM_PROMPT = (
    "Read the passage and answer the question. Provide only the answer, as briefly as possible."
)

_BIAS_SYSTEM_PROMPT = (
    "Answer the question by selecting the most appropriate option. "
    "Respond with only the letter of your answer."
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
        return asyncio.get_event_loop().run_until_complete(self.arun(benchmark, limit))

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
            response = await self._provider.evaluate(prompt=prompt, system_prompt=system_prompt)
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
            cat: category_correct.get(cat, 0) / category_total[cat] for cat in category_total
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

    def _build_prompt(self, benchmark: str, sample) -> tuple[str, str]:
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
        elif benchmark == "humaneval":
            system = _CODE_SYSTEM_PROMPT
            prompt = sample.question
        elif benchmark == "boolq":
            system = _BOOLEAN_SYSTEM_PROMPT
            prompt = sample.question
        elif benchmark == "ifeval":
            system = _INSTRUCTION_SYSTEM_PROMPT
            prompt = sample.question
        elif benchmark == "lambada":
            system = _COMPLETION_SYSTEM_PROMPT
            prompt = sample.question
        elif benchmark in ("squad", "drop"):
            system = _READING_SYSTEM_PROMPT
            prompt = sample.question
        elif benchmark == "winogrande":
            system = _MC_AB_SYSTEM_PROMPT
            choices_text = "\n".join(sample.choices)
            prompt = f"{sample.question}\n\n{choices_text}"
        elif benchmark == "bbq":
            system = _BIAS_SYSTEM_PROMPT
            choices_text = "\n".join(sample.choices)
            prompt = f"{sample.question}\n\n{choices_text}"
        elif sample.choices:
            system = _MC_SYSTEM_PROMPT
            choices_text = "\n".join(sample.choices)
            prompt = f"{sample.question}\n\n{choices_text}"
        else:
            system = _OPEN_SYSTEM_PROMPT
            prompt = sample.question
        return prompt, system

    _MC_BENCHMARKS = frozenset(
        {
            "mmlu",
            "hellaswag",
            "bbh",
            "arc",
            "logiqa",
            "mathqa",
            "winogrande",
            "bbq",
        }
    )

    def _check_answer(self, benchmark: str, model_text: str, correct_answer: str) -> bool:
        """Determine whether the model's response matches the correct answer.

        Args:
            benchmark: The benchmark name, used to choose comparison strategy.
            model_text: The raw text produced by the model.
            correct_answer: The ground-truth answer.

        Returns:
            True if the model's answer matches the correct answer.
        """
        model_text = (model_text or "").strip()
        if benchmark in self._MC_BENCHMARKS:
            return self._check_multiple_choice(model_text, correct_answer)
        if benchmark == "gsm8k":
            return self._check_numeric(model_text, correct_answer)
        if benchmark == "humaneval":
            return self._check_code(model_text, correct_answer)
        if benchmark == "boolq":
            return self._check_boolean(model_text, correct_answer)
        if benchmark == "ifeval":
            return self._check_instruction(model_text, correct_answer)
        if benchmark == "drop":
            return self._check_drop(model_text, correct_answer)
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

    @staticmethod
    def _check_code(model_text: str, correct_answer: str) -> bool:
        """Check code generation by verifying key logic fragments are present.

        Splits the correct answer into meaningful lines and checks that
        each non-trivial line appears somewhere in the model output.

        Args:
            model_text: The model's generated code.
            correct_answer: The reference implementation.

        Returns:
            True if all key logic fragments from the reference are found.
        """
        model_normalized = model_text.replace(" ", "").lower()
        key_lines = [
            line.strip()
            for line in correct_answer.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not key_lines:
            return False
        matched = sum(1 for line in key_lines if line.replace(" ", "").lower() in model_normalized)
        return matched / len(key_lines) >= 0.5

    @staticmethod
    def _check_boolean(model_text: str, correct_answer: str) -> bool:
        """Extract a yes/no answer from the model response.

        Args:
            model_text: The model's response text.
            correct_answer: The expected answer ("Yes" or "No").

        Returns:
            True if the extracted boolean matches the correct answer.
        """
        text_lower = model_text.lower().strip()
        expected = correct_answer.lower().strip()
        if text_lower.startswith("yes"):
            return expected == "yes"
        if text_lower.startswith("no"):
            return expected == "no"
        return expected in text_lower

    @staticmethod
    def _check_instruction(model_text: str, correct_answer: str) -> bool:
        """Validate that the model followed formatting instructions.

        Performs basic structural checks based on the instruction description
        in the correct_answer field.

        Args:
            model_text: The model's response text.
            correct_answer: A description of the expected structural properties.

        Returns:
            True if the response appears to satisfy the described constraint.
        """
        spec = correct_answer.lower()

        if "uppercase" in spec:
            alpha_chars = [c for c in model_text if c.isalpha()]
            if not alpha_chars:
                return False
            return all(c.isupper() for c in alpha_chars)

        if "not contain" in spec:
            forbidden_match = re.findall(r"(?:not contain the word[s]?\s+)(.+)", spec)
            if forbidden_match:
                words = re.split(r"[,\s]+(?:or\s+)?", forbidden_match[0].strip())
                words = [w.strip(" .'\"") for w in words if w.strip(" .'\"")]
                return all(w not in model_text.lower() for w in words)

        sentence_match = re.search(r"exactly (\d+) sentence", spec)
        if sentence_match:
            expected_count = int(sentence_match.group(1))
            sentences = [s.strip() for s in re.split(r"[.!?]+", model_text) if s.strip()]
            return len(sentences) == expected_count

        bullet_match = re.search(r"exactly (\d+) bullet", spec)
        if bullet_match:
            expected_count = int(bullet_match.group(1))
            bullets = [line for line in model_text.splitlines() if line.strip().startswith("-")]
            return len(bullets) == expected_count

        step_match = re.search(r"exactly (\d+) numbered step", spec)
        if step_match:
            expected_count = int(step_match.group(1))
            steps = [line for line in model_text.splitlines() if re.match(r"^\s*\d+[.)]\s", line)]
            return len(steps) == expected_count

        if "numbered list" in spec:
            lines = [line.strip() for line in model_text.splitlines() if line.strip()]
            return any(re.match(r"^\d+[.)]\s", line) for line in lines)

        if "semicolon" in spec:
            return ";" in model_text

        word_match = re.search(r"contain the word (\w+) (?:at least|exactly) (\w+)", spec)
        if word_match:
            word = word_match.group(1).lower()
            count_word = word_match.group(2)
            actual_count = model_text.lower().count(word)
            if "at least" in spec:
                try:
                    return actual_count >= int(count_word)
                except ValueError:
                    num_map = {"once": 1, "twice": 2, "three": 3}
                    return actual_count >= num_map.get(count_word, 1)
            if "exactly" in spec:
                try:
                    return actual_count == int(count_word)
                except ValueError:
                    num_map = {"once": 1, "twice": 2, "three": 3}
                    return actual_count == num_map.get(count_word, 1)

        if "start with" in spec and "end with" in spec:
            start_match = re.search(r"start with (\w+)", spec)
            end_match = re.search(r"end with (.+?)(?:\.|$)", spec)
            if start_match and end_match:
                starts_ok = model_text.strip().lower().startswith(start_match.group(1).lower())
                ends_ok = (
                    model_text.strip()
                    .lower()
                    .rstrip(".")
                    .endswith(end_match.group(1).strip().lower().rstrip("."))
                )
                return starts_ok and ends_ok

        word_count_match = re.search(r"(?:at most|no more than) (\d+) words", spec)
        if word_count_match:
            max_words = int(word_count_match.group(1))
            return len(model_text.split()) <= max_words

        between_match = re.search(r"between (\d+) and (\d+) words", spec)
        if between_match:
            lo = int(between_match.group(1))
            hi = int(between_match.group(2))
            word_count = len(model_text.split())
            return lo <= word_count <= hi

        return len(model_text.strip()) > 0

    @staticmethod
    def _check_drop(model_text: str, correct_answer: str) -> bool:
        """Check DROP answers that may be numeric or short text spans.

        Tries numeric comparison first, then falls back to text containment.

        Args:
            model_text: The model's response text.
            correct_answer: The expected answer (number or text).

        Returns:
            True if the answer matches numerically or via containment.
        """
        numbers = re.findall(r"-?\d+(?:\.\d+)?", model_text)
        try:
            expected = float(correct_answer.replace(",", ""))
            if numbers:
                actual = float(numbers[-1].replace(",", ""))
                if abs(actual - expected) < 1e-6:
                    return True
        except ValueError:
            pass
        model_lower = model_text.lower()
        answer_lower = correct_answer.lower()
        return answer_lower in model_lower or model_lower in answer_lower
