"""Dashboard cloud sync for checkllm.

Pushes the local SQLite run history to a remote HTTP collector. Useful
for fan-in scenarios where many CI workers produce runs that should be
aggregated on a central dashboard.

Usage::

    from checkllm.history import RunHistory
    from checkllm.integrations.cloud_sync import push_to_remote

    history = RunHistory()
    push_to_remote(history, url="https://collector.example.com/runs")

Environment variables:
    ``CHECKLLM_REMOTE_URL`` — default remote endpoint.
    ``CHECKLLM_REMOTE_TOKEN`` — bearer token for authentication.

This module has no required third-party dependencies. ``urllib`` from
the standard library is used by default; when the ``requests`` package
is installed it is used automatically for connection pooling.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from checkllm.history import RunHistory

logger = logging.getLogger("checkllm.integrations.cloud_sync")


@dataclass
class SyncResult:
    """Result of a ``push_to_remote`` invocation.

    Attributes:
        runs_pushed: Number of runs that were serialized and sent.
        status_code: HTTP status code returned by the collector.
        ok: Whether the push is considered successful.
        error: Error description when ``ok`` is false.
    """

    runs_pushed: int
    status_code: int = 0
    ok: bool = False
    error: str | None = None


def _serialize_runs(history: RunHistory, limit: int) -> list[dict[str, Any]]:
    """Serialize recent runs into JSON-ready dicts.

    Args:
        history: Source run history store.
        limit: Maximum number of most recent runs to include.

    Returns:
        A list of dicts describing each run.
    """
    runs: list[dict[str, Any]] = []
    for summary in history.list_runs(limit=limit):
        record = history.get_run(summary.run_id)
        if record is None:
            continue
        runs.append(asdict(record))
    return runs


def _post_json(
    url: str, payload: dict[str, Any], token: str | None, timeout: float
) -> tuple[int, str]:
    """POST JSON to ``url`` and return ``(status_code, body)``.

    Args:
        url: Target endpoint.
        payload: JSON-serializable dict to send.
        token: Optional bearer token.
        timeout: Request timeout in seconds.

    Returns:
        A tuple of the HTTP status code and the decoded response body.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import requests  # type: ignore[import-untyped,unused-ignore]

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        status_code: int = response.status_code
        text: str = response.text
        return status_code, text
    except ImportError:
        pass

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body


def push_to_remote(
    history: RunHistory,
    url: str | None = None,
    token: str | None = None,
    limit: int = 50,
    timeout: float = 10.0,
    extra: dict[str, Any] | None = None,
) -> SyncResult:
    """Push aggregated run history to a remote collector.

    Args:
        history: The :class:`checkllm.history.RunHistory` to drain.
        url: Collector URL. Falls back to ``CHECKLLM_REMOTE_URL``.
        token: Bearer token. Falls back to ``CHECKLLM_REMOTE_TOKEN``.
        limit: Maximum number of recent runs to ship.
        timeout: HTTP timeout in seconds.
        extra: Optional additional metadata embedded at the top level of
            the payload (e.g. ``{"source": "ci-worker-3"}``).

    Returns:
        A :class:`SyncResult` describing the outcome.

    Raises:
        ValueError: If neither ``url`` nor ``CHECKLLM_REMOTE_URL`` is set.
    """
    resolved_url = url or os.getenv("CHECKLLM_REMOTE_URL")
    if not resolved_url:
        raise ValueError("Remote URL not provided; set CHECKLLM_REMOTE_URL or pass url=...")
    resolved_token = token or os.getenv("CHECKLLM_REMOTE_TOKEN")

    runs = _serialize_runs(history, limit=limit)
    payload: dict[str, Any] = {"runs": runs}
    if extra:
        payload.update(extra)

    try:
        status, body = _post_json(resolved_url, payload, resolved_token, timeout)
    except Exception as exc:
        logger.warning("cloud sync failed: %s", exc)
        return SyncResult(runs_pushed=len(runs), status_code=0, ok=False, error=str(exc))

    ok = 200 <= status < 300
    if not ok:
        logger.warning("cloud sync returned %d: %s", status, body[:500])
    return SyncResult(
        runs_pushed=len(runs),
        status_code=status,
        ok=ok,
        error=None if ok else body[:500],
    )
