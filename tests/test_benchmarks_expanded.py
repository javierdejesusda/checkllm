"""Expanded tests for the 16-benchmark suite in checkllm.benchmarks."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from checkllm.benchmarks.datasets import (
    BenchmarkDataset,
    BenchmarkSample,
    list_benchmarks,
    load_benchmark,
)
from checkllm.benchmarks.runner import BenchmarkRunner


ALL_BENCHMARKS = [
    "arc",
    "bbh",
    "bbq",
    "boolq",
    "drop",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "ifeval",
    "lambada",
    "logiqa",
    "mathqa",
    "mmlu",
    "squad",
    "truthfulqa",
    "winogrande",
]


def test_list_benchmarks_returns_all_16():
    names = list_benchmarks()
    assert len(names) == 16
    assert names == ALL_BENCHMARKS


def test_list_benchmarks_is_sorted():
    names = list_benchmarks()
    assert names == sorted(names)


@pytest.mark.parametrize("name", ALL_BENCHMARKS)
def test_load_benchmark_succeeds(name):
    dataset = load_benchmark(name)
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.name == name
    assert len(dataset.samples) >= 15


@pytest.mark.parametrize("name", ALL_BENCHMARKS)
def test_all_samples_have_required_fields(name):
    dataset = load_benchmark(name)
    for sample in dataset.samples:
        assert isinstance(sample, BenchmarkSample)
        assert sample.question
        assert sample.correct_answer
        assert sample.category


@pytest.mark.parametrize("name", ALL_BENCHMARKS)
def test_benchmark_limit(name):
    dataset = load_benchmark(name, limit=3)
    assert len(dataset.samples) == 3


MC_BENCHMARKS = [
    "mmlu", "hellaswag", "bbh", "arc", "logiqa", "mathqa", "winogrande", "bbq",
]


@pytest.mark.parametrize("name", MC_BENCHMARKS)
def test_mc_benchmarks_have_choices(name):
    dataset = load_benchmark(name)
    for sample in dataset.samples:
        assert sample.choices is not None
        assert len(sample.choices) >= 2


OPEN_BENCHMARKS = [
    "truthfulqa", "gsm8k", "humaneval", "boolq", "drop", "ifeval",
    "lambada", "squad",
]


@pytest.mark.parametrize("name", OPEN_BENCHMARKS)
def test_open_benchmarks_have_no_choices(name):
    dataset = load_benchmark(name)
    for sample in dataset.samples:
        assert sample.choices is None


@pytest.mark.parametrize("name", ALL_BENCHMARKS)
def test_each_benchmark_has_multiple_categories(name):
    dataset = load_benchmark(name)
    categories = {s.category for s in dataset.samples}
    assert len(categories) >= 2, (
        f"{name} only has categories: {categories}"
    )


def test_winogrande_has_two_choices():
    dataset = load_benchmark("winogrande")
    for sample in dataset.samples:
        assert len(sample.choices) == 2


def test_bbq_has_three_choices():
    dataset = load_benchmark("bbq")
    for sample in dataset.samples:
        assert len(sample.choices) == 3


def test_boolq_answers_are_yes_or_no():
    dataset = load_benchmark("boolq")
    for sample in dataset.samples:
        assert sample.correct_answer in ("Yes", "No")


class TestCheckCode:
    """Tests for the _check_code answer checker."""

    def setup_method(self):
        self.runner = BenchmarkRunner(provider=MagicMock())

    def test_exact_match(self):
        correct = "return s[::-1]"
        model = "return s[::-1]"
        assert self.runner._check_code(model, correct)

    def test_partial_match_above_threshold(self):
        correct = (
            "seen = {}\n"
            "for i, num in enumerate(nums):\n"
            "    complement = target - num\n"
            "    if complement in seen:\n"
            "        return [seen[complement], i]\n"
            "    seen[num] = i"
        )
        model = (
            "seen = {}\n"
            "for i, num in enumerate(nums):\n"
            "    complement = target - num\n"
            "    if complement in seen:\n"
            "        return [seen[complement], i]\n"
            "    seen[num] = i\n"
            "return []"
        )
        assert self.runner._check_code(model, correct)

    def test_no_match(self):
        correct = "return s[::-1]"
        model = "print('hello world')"
        assert not self.runner._check_code(model, correct)


class TestCheckBoolean:
    """Tests for the _check_boolean answer checker."""

    def setup_method(self):
        self.runner = BenchmarkRunner(provider=MagicMock())

    def test_yes_matches_yes(self):
        assert self.runner._check_boolean("Yes", "Yes")

    def test_no_matches_no(self):
        assert self.runner._check_boolean("No", "No")

    def test_yes_with_explanation(self):
        assert self.runner._check_boolean("Yes, that is correct.", "Yes")

    def test_no_with_explanation(self):
        assert self.runner._check_boolean("No, that is incorrect.", "No")

    def test_wrong_answer(self):
        assert not self.runner._check_boolean("Yes", "No")

    def test_case_insensitive(self):
        assert self.runner._check_boolean("yes", "Yes")


class TestCheckInstruction:
    """Tests for the _check_instruction answer checker."""

    def setup_method(self):
        self.runner = BenchmarkRunner(provider=MagicMock())

    def test_sentence_count(self):
        spec = "response should contain exactly 3 sentences"
        text = "First sentence. Second sentence. Third sentence."
        assert self.runner._check_instruction(text, spec)

    def test_sentence_count_wrong(self):
        spec = "response should contain exactly 3 sentences"
        text = "One sentence. Two sentences."
        assert not self.runner._check_instruction(text, spec)

    def test_uppercase(self):
        spec = "response should be entirely in uppercase"
        assert self.runner._check_instruction("THIS IS UPPERCASE", spec)

    def test_uppercase_fails(self):
        spec = "response should be entirely in uppercase"
        assert not self.runner._check_instruction("This is Mixed", spec)

    def test_not_contain_word(self):
        spec = "response should not contain the word data"
        assert self.runner._check_instruction("Machine learning uses patterns.", spec)

    def test_not_contain_word_fails(self):
        spec = "response should not contain the word data"
        assert not self.runner._check_instruction("ML uses data.", spec)

    def test_bullet_points(self):
        spec = "response should contain exactly 3 bullet points starting with -"
        text = "- Tip one\n- Tip two\n- Tip three"
        assert self.runner._check_instruction(text, spec)

    def test_semicolons(self):
        spec = "response should use semicolons as separators"
        text = "Python; JavaScript; Go; Rust; C++"
        assert self.runner._check_instruction(text, spec)

    def test_word_count_limit(self):
        spec = "response should be one sentence with at most 25 words"
        text = "The sun is a star."
        assert self.runner._check_instruction(text, spec)

    def test_numbered_list(self):
        spec = "response should be a numbered list with items starting with digits"
        text = "1. Mercury\n2. Venus\n3. Earth"
        assert self.runner._check_instruction(text, spec)


class TestCheckDrop:
    """Tests for the _check_drop answer checker."""

    def setup_method(self):
        self.runner = BenchmarkRunner(provider=MagicMock())

    def test_numeric_answer(self):
        assert self.runner._check_drop("The answer is 34.", "34")

    def test_text_answer(self):
        assert self.runner._check_drop(
            "Team A won by 0 points; the game was tied 31-31",
            "Team A won by 0 points; the game was tied 31-31",
        )

    def test_wrong_numeric(self):
        assert not self.runner._check_drop("The answer is 50.", "34")


class TestBuildPrompt:
    """Tests for _build_prompt routing to correct system prompts."""

    def setup_method(self):
        self.runner = BenchmarkRunner(provider=MagicMock())

    def _make_sample(self, question="Q?", choices=None, answer="A"):
        return BenchmarkSample(
            question=question,
            choices=choices,
            correct_answer=answer,
            category="test",
        )

    def test_humaneval_uses_code_prompt(self):
        sample = self._make_sample(question="def foo():\n    pass")
        _, system = self.runner._build_prompt("humaneval", sample)
        assert "function body" in system.lower() or "implementation" in system.lower()

    def test_boolq_uses_boolean_prompt(self):
        sample = self._make_sample(question="Is the sky blue?")
        _, system = self.runner._build_prompt("boolq", sample)
        assert "yes" in system.lower() or "no" in system.lower()

    def test_lambada_uses_completion_prompt(self):
        sample = self._make_sample(question="The cat sat on the")
        _, system = self.runner._build_prompt("lambada", sample)
        assert "word" in system.lower() or "complete" in system.lower()

    def test_squad_uses_reading_prompt(self):
        sample = self._make_sample(question="Passage: ... Question: ?")
        _, system = self.runner._build_prompt("squad", sample)
        assert "passage" in system.lower() or "read" in system.lower()

    def test_winogrande_uses_ab_prompt(self):
        sample = self._make_sample(
            choices=["A. Option 1", "B. Option 2"], answer="A"
        )
        _, system = self.runner._build_prompt("winogrande", sample)
        assert "a or b" in system.lower() or "two options" in system.lower()

    def test_ifeval_uses_instruction_prompt(self):
        sample = self._make_sample(question="Write in uppercase.")
        _, system = self.runner._build_prompt("ifeval", sample)
        assert "instruction" in system.lower() or "follow" in system.lower()


@pytest.mark.asyncio
async def test_runner_processes_each_benchmark():
    """Verify BenchmarkRunner can process a sample from every benchmark."""
    mock_response = MagicMock()
    mock_response.cost = 0.0
    mock_response.raw_output = "B"
    mock_response.reasoning = "B"

    provider = MagicMock()
    provider.evaluate = AsyncMock(return_value=mock_response)

    runner = BenchmarkRunner(provider=provider)

    for name in ALL_BENCHMARKS:
        result = await runner.arun(name, limit=1)
        assert result.benchmark == name
        assert result.total == 1


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("nonexistent_benchmark")


def test_case_insensitive_loading():
    for name in ["MMLU", "HellaSwag", "BoolQ", "SQUAD"]:
        dataset = load_benchmark(name, limit=1)
        assert dataset.name == name.lower()
