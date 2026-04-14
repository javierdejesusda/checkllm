import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def tiny_halubench() -> list[dict]:
    return _load("halubench_tiny.json")


@pytest.fixture
def tiny_ragtruth() -> list[dict]:
    return _load("ragtruth_tiny.json")


@pytest.fixture
def tiny_truthfulqa() -> list[dict]:
    return _load("truthfulqa_tiny.json")


@pytest.fixture
def tiny_jailbreakbench() -> list[dict]:
    return _load("jailbreakbench_tiny.json")
