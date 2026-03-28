from pathlib import Path

import pytest

from checkllm.datasets.case import Case
from checkllm.datasets.loader import load_yaml_dataset, load_dataset


class TestCase:
    def test_create_case_with_all_fields(self):
        case = Case(
            input="test input",
            query="test query",
            criteria="test criteria",
            metadata={"key": "value"},
        )
        assert case.input == "test input"
        assert case.query == "test query"
        assert case.criteria == "test criteria"
        assert case.metadata == {"key": "value"}

    def test_create_case_with_minimal_fields(self):
        case = Case(input="test input")
        assert case.input == "test input"
        assert case.query is None
        assert case.criteria is None
        assert case.metadata == {}

    def test_case_with_expected_and_context(self):
        case = Case(
            input="What is 2+2?",
            expected="4",
            context="Math problem",
        )
        assert case.expected == "4"
        assert case.context == "Math problem"

    def test_case_expected_criteria_alias(self):
        case = Case(input="test", criteria="must be concise")
        assert case.expected_criteria == "must be concise"


class TestLoadYamlDataset:
    def test_load_sample_dataset(self):
        path = Path(__file__).parent / "fixtures" / "sample_dataset.yaml"
        cases = load_yaml_dataset(path)
        assert len(cases) == 2
        assert cases[0].input == "The Earth orbits the Sun in approximately 365.25 days."
        assert cases[0].query == "How long does Earth take to orbit the Sun?"
        assert cases[0].criteria == "mentions 365 days, is concise"

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_dataset(Path("/nonexistent/file.yaml"))


class TestLoadDataset:
    def test_load_yaml_by_extension(self):
        path = Path(__file__).parent / "fixtures" / "sample_dataset.yaml"
        cases = load_dataset(path)
        assert len(cases) == 2
        assert all(isinstance(c, Case) for c in cases)

    def test_load_from_generator_function(self):
        def my_cases():
            yield Case(input="case 1")
            yield Case(input="case 2")
            yield Case(input="case 3")

        cases = load_dataset(my_cases)
        assert len(cases) == 3
        assert cases[2].input == "case 3"
