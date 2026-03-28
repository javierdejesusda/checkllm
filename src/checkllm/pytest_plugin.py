from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

import pytest

from checkllm.check import CheckCollector
from checkllm.config import load_config
from checkllm.datasets.case import Case
from checkllm.datasets.loader import load_dataset

_CHECKLLM_KEY = pytest.StashKey[CheckCollector]()


@pytest.fixture
def check(request):
    """Provides a CheckCollector that collects results and raises on teardown."""
    config = load_config()
    collector = CheckCollector(config=config)
    request.node.stash[_CHECKLLM_KEY] = collector
    return collector


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail the test at call-report time if check collector has failures."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        collector = item.stash.get(_CHECKLLM_KEY, None)
        if collector is not None:
            failed = [r for r in collector.results if not r.passed]
            if failed:
                count = len(failed)
                names = ", ".join(r.metric_name for r in failed)
                report.outcome = "failed"
                report.longrepr = f"{count} check(s) failed: {names}"


def dataset(source: str | Path | Callable) -> Callable:
    """Decorator that parametrizes a test function over a dataset.

    Usage::

        @dataset("tests/datasets/cases.yaml")
        def test_my_agent(check, case):
            result = my_agent(case.input)
            check.hallucination(result, context=case.input)
    """

    def decorator(func: Callable) -> Callable:
        cases = load_dataset(source)
        ids = [
            f"case-{i}-{c.query or c.input[:30]}"
            for i, c in enumerate(cases)
        ]

        @pytest.mark.parametrize("case", cases, ids=ids)
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator
