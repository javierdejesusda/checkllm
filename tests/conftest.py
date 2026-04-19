pytest_plugins = ["pytester"]

import pytest
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig


@pytest.fixture
def check():
    """Fallback check fixture for tests that don't load the checkllm plugin."""
    config = CheckllmConfig(cache_enabled=False)
    return CheckCollector(config=config)
