"""CI platform helpers for checkllm.

This package contains platform-specific helpers used by the
``checkllm ci`` subcommand. Each module exposes a small, stable API:

* ``detect()`` — returns True when the platform is detected.
* ``post_comment(...)`` — posts a comment to a merge request / PR.
* ``template()`` — returns a ready-to-paste pipeline YAML snippet.

The existing GitHub Actions logic continues to live in
``checkllm.cicd`` for backwards compatibility; this package adds
parallel helpers without moving files.
"""

from __future__ import annotations

from checkllm.ci import gitlab

__all__ = ["gitlab"]
