"""Tests for the dashboard cloud-sync helper."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from checkllm.history import RunHistory
from checkllm.integrations import cloud_sync
from checkllm.models import CheckResult


@pytest.fixture
def history(tmp_path):
    store = RunHistory(db_path=tmp_path / "history.db")
    yield store
    store.close()


def _seed(history: RunHistory) -> None:
    checks = {
        "test_one": [
            CheckResult(
                passed=True,
                score=0.9,
                reasoning="good",
                cost=0.01,
                latency_ms=10,
                metric_name="relevance",
            )
        ]
    }
    history.record_run(checks, label="ci")


def test_push_requires_url(history):
    with pytest.raises(ValueError):
        cloud_sync.push_to_remote(history, url=None)


def test_push_sends_runs(monkeypatch, history):
    _seed(history)
    captured: dict = {}

    def fake_post(url, payload, token, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["token"] = token
        captured["timeout"] = timeout
        return 200, "{}"

    monkeypatch.setattr(cloud_sync, "_post_json", fake_post)
    result = cloud_sync.push_to_remote(
        history, url="http://collector.test/runs", token="t"
    )
    assert result.ok is True
    assert result.status_code == 200
    assert result.runs_pushed == 1
    assert captured["url"] == "http://collector.test/runs"
    assert captured["token"] == "t"
    assert captured["payload"]["runs"][0]["label"] == "ci"


def test_push_reports_http_failure(monkeypatch, history):
    _seed(history)

    def fake_post(url, payload, token, timeout):
        return 500, "bad"

    monkeypatch.setattr(cloud_sync, "_post_json", fake_post)
    result = cloud_sync.push_to_remote(history, url="http://x/y")
    assert result.ok is False
    assert result.status_code == 500
    assert result.error == "bad"


def test_push_catches_exceptions(monkeypatch, history):
    _seed(history)

    def boom(url, payload, token, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(cloud_sync, "_post_json", boom)
    result = cloud_sync.push_to_remote(history, url="http://x/y")
    assert result.ok is False
    assert result.error == "network down"


def test_env_var_url_fallback(monkeypatch, history):
    _seed(history)
    monkeypatch.setenv("CHECKLLM_REMOTE_URL", "http://env-remote/runs")
    monkeypatch.setenv("CHECKLLM_REMOTE_TOKEN", "envtoken")
    seen: dict = {}

    def fake_post(url, payload, token, timeout):
        seen["url"] = url
        seen["token"] = token
        return 201, ""

    monkeypatch.setattr(cloud_sync, "_post_json", fake_post)
    cloud_sync.push_to_remote(history)
    assert seen["url"] == "http://env-remote/runs"
    assert seen["token"] == "envtoken"


def test_extra_metadata_is_merged(monkeypatch, history):
    _seed(history)
    seen: dict = {}

    def fake_post(url, payload, token, timeout):
        seen["payload"] = payload
        return 200, ""

    monkeypatch.setattr(cloud_sync, "_post_json", fake_post)
    cloud_sync.push_to_remote(
        history, url="http://x/y", extra={"source": "ci-worker-3"}
    )
    assert seen["payload"]["source"] == "ci-worker-3"
    assert "runs" in seen["payload"]


def test_post_json_uses_requests_when_available(monkeypatch):
    """_post_json should return status/body from requests when importable."""
    import sys
    import types

    fake_resp = MagicMock()
    fake_resp.status_code = 201
    fake_resp.text = '{"id": 1}'

    fake_requests = types.ModuleType("requests")
    fake_requests.post = MagicMock(return_value=fake_resp)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    status, body = cloud_sync._post_json(
        "http://collector.test/ingest",
        {"runs": []},
        token="tok",
        timeout=5.0,
    )
    assert status == 201
    assert body == '{"id": 1}'
    fake_requests.post.assert_called_once()
    kwargs = fake_requests.post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


def test_post_json_falls_back_to_urllib(monkeypatch):
    """When requests cannot be imported, _post_json uses urllib."""
    import sys

    # Make `import requests` raise ImportError inside the helper.
    monkeypatch.setitem(sys.modules, "requests", None)

    captured: dict = {}

    class FakeResp:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", fake_urlopen)
    status, body = cloud_sync._post_json(
        "http://collector.test/x",
        {"runs": [{"id": 1}]},
        token=None,
        timeout=3.0,
    )
    assert status == 202
    assert "ok" in body
    assert captured["method"] == "POST"
    assert json.loads(captured["data"].decode("utf-8"))["runs"][0]["id"] == 1


def test_post_json_surfaces_http_error(monkeypatch):
    """HTTPError should be unwrapped into (code, body)."""
    import io
    import sys

    monkeypatch.setitem(sys.modules, "requests", None)

    def fake_urlopen(req, timeout):
        raise cloud_sync.urllib.error.HTTPError(
            url=req.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"overloaded"),
        )

    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", fake_urlopen)
    status, body = cloud_sync._post_json(
        "http://x", {"a": 1}, token=None, timeout=2.0
    )
    assert status == 503
    assert body == "overloaded"


def test_push_uses_resolved_runs(monkeypatch, history):
    """push_to_remote should gracefully handle an empty history."""
    captured: dict = {}

    def fake_post(url, payload, token, timeout):
        captured["payload"] = payload
        return 200, "{}"

    monkeypatch.setattr(cloud_sync, "_post_json", fake_post)
    result = cloud_sync.push_to_remote(history, url="http://x/y")
    assert result.ok is True
    assert result.runs_pushed == 0
    assert captured["payload"]["runs"] == []
