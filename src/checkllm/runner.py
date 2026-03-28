from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_pytest(
    test_path: str,
    junit_xml: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run pytest on the given test path and return the exit code."""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    if junit_xml:
        cmd.append(f"--junit-xml={junit_xml}")
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd)
    return result.returncode
