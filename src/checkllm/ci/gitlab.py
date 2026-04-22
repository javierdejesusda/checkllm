"""GitLab CI helpers for the ``checkllm ci`` subcommand.

This module mirrors the GitHub Actions auto-detection used by
``checkllm.cicd.github_action``:

* :func:`detect` returns True when the current process is running
  inside a GitLab CI job.
* :func:`post_mr_comment` posts a note to the triggering merge
  request via the GitLab REST API.
* :func:`gitlab_template` returns a minimal ``.gitlab-ci.yml``
  snippet users can paste into their project.
"""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class GitLabContext:
    """Describes the current GitLab CI environment.

    Attributes:
        server_url: Base URL of the GitLab instance (``CI_SERVER_URL``).
        project_id: Numeric project identifier (``CI_PROJECT_ID``).
        mr_iid: Merge request internal ID, when the pipeline was
            triggered by an MR. ``None`` for branch pipelines.
        token: Token used for API authentication. Prefers an explicit
            personal / project access token, falling back to
            ``CI_JOB_TOKEN``.
        token_is_job_token: True when the fallback ``CI_JOB_TOKEN`` is
            the only credential available. Job tokens cannot post
            notes on most GitLab instances, so callers should warn
            the user.
    """

    server_url: str
    project_id: str
    mr_iid: Optional[str]
    token: Optional[str]
    token_is_job_token: bool


def detect() -> bool:
    """Return True when running inside a GitLab CI job.

    Detection is intentionally conservative: we require ``GITLAB_CI``
    to be truthy *and* ``CI_PROJECT_ID`` to be set. Either alone can
    be spoofed by unrelated tooling.

    Returns:
        True when GitLab CI is detected.
    """
    if os.environ.get("GITLAB_CI", "").lower() != "true":
        return False
    return bool(os.environ.get("CI_PROJECT_ID"))


def context_from_env() -> Optional[GitLabContext]:
    """Build a :class:`GitLabContext` from the current environment.

    Returns:
        A populated :class:`GitLabContext`, or ``None`` when the
        process is not running inside GitLab CI.
    """
    if not detect():
        return None

    server_url = os.environ.get("CI_SERVER_URL", "https://gitlab.com")
    project_id = os.environ.get("CI_PROJECT_ID", "")
    mr_iid = os.environ.get("CI_MERGE_REQUEST_IID") or None

    explicit_token = (
        os.environ.get("CHECKLLM_GITLAB_TOKEN")
        or os.environ.get("GITLAB_TOKEN")
        or os.environ.get("GITLAB_API_TOKEN")
    )
    job_token = os.environ.get("CI_JOB_TOKEN")

    token = explicit_token or job_token
    token_is_job_token = bool(job_token) and not explicit_token

    return GitLabContext(
        server_url=server_url.rstrip("/"),
        project_id=project_id,
        mr_iid=mr_iid,
        token=token,
        token_is_job_token=token_is_job_token,
    )


def post_mr_comment(
    body: str,
    *,
    ctx: Optional[GitLabContext] = None,
    timeout: float = 10.0,
) -> bool:
    """Post a comment to the current merge request.

    Args:
        body: Markdown body of the note.
        ctx: Optional :class:`GitLabContext`. When omitted the
            context is read from the environment.
        timeout: HTTP timeout in seconds.

    Returns:
        True when the comment was posted successfully, False when
        posting was skipped (missing context / missing MR / token
        error). Skips are silent — callers can log their own
        diagnostics.
    """
    if ctx is None:
        ctx = context_from_env()
    if ctx is None or ctx.mr_iid is None or not ctx.token:
        return False

    url = (
        f"{ctx.server_url}/api/v4/projects/{ctx.project_id}"
        f"/merge_requests/{ctx.mr_iid}/notes"
    )
    payload = json.dumps({"body": body}).encode("utf-8")

    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if ctx.token_is_job_token:
        req.add_header("JOB-TOKEN", ctx.token)
    else:
        req.add_header("PRIVATE-TOKEN", ctx.token)

    try:
        with urlopen(req, timeout=timeout) as resp:
            status: int = resp.status
            return 200 <= status < 300
    except (HTTPError, URLError, OSError):
        return False


def gitlab_template(
    *,
    eval_command: str = "checkllm ci tests/",
    python_version: str = "3.11",
    budget: Optional[float] = None,
) -> str:
    """Return a minimal ``.gitlab-ci.yml`` snippet.

    The snippet defines a single job that installs checkllm, runs the
    evaluation command, and uploads the JSON report as a pipeline
    artifact. It is safe to paste into an existing pipeline.

    Args:
        eval_command: Shell command that runs the evaluation. The
            default uses ``checkllm ci`` so MR comments are posted
            automatically.
        python_version: Python version tag of the Docker image.
        budget: Optional maximum USD spend appended to the command.

    Returns:
        A valid YAML string.
    """
    cmd = eval_command
    if budget is not None:
        cmd = f"{eval_command} --budget {budget:.2f}"

    return textwrap.dedent(
        f"""\
        checkllm:
          image: python:{python_version}-slim
          stage: test
          before_script:
            - python -m pip install --upgrade pip
            - pip install checkllm
          script:
            - {cmd}
          artifacts:
            when: always
            paths:
              - .checkllm/ci_snapshot.json
            expire_in: 30 days
          rules:
            - if: $CI_PIPELINE_SOURCE == "merge_request_event"
            - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
        """
    )


def format_mr_comment(results: Mapping[str, Iterable[Any]]) -> str:
    """Render a short Markdown summary from a results dictionary.

    The dictionary is produced by the pytest plugin snapshot loader
    and follows the shape ``{test_id: [CheckResult, ...]}``. Only
    aggregate counts are emitted so the helper keeps zero direct
    dependencies on the CheckResult class.

    Args:
        results: Mapping of test ids to iterables of objects with a
            ``passed`` attribute.

    Returns:
        A Markdown string suitable for ``post_mr_comment``.
    """
    total = 0
    passed = 0
    for checks in results.values():
        for check in checks:
            total += 1
            if getattr(check, "passed", False):
                passed += 1

    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0

    lines = [
        "## checkllm Evaluation Results",
        "",
        f"- **Tests:** {len(results)}",
        f"- **Checks:** {total}",
        f"- **Passed:** {passed}",
        f"- **Failed:** {failed}",
        f"- **Pass rate:** {pass_rate:.1f}%",
    ]
    return "\n".join(lines) + "\n"
