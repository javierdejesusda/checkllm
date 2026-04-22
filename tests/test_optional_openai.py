"""Tests verifying that core checkllm functionality works without openai installed."""

import subprocess
import sys


def test_deterministic_checks_import_without_openai():
    """DeterministicChecks.contains() runs in a subprocess that cannot import openai."""
    script = (
        "import sys; sys.modules['openai'] = None; "
        "from checkllm.deterministic import DeterministicChecks; "
        "dc = DeterministicChecks(); "
        "result = dc.contains('hello world', 'hello'); "
        "assert result.passed is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed with stderr:\n{result.stderr}"


def test_checkllm_version_accessible():
    """__version__ is importable from checkllm without errors."""
    from checkllm import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0
