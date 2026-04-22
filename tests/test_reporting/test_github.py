"""Tests for GitHub integration module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from checkllm.models import CheckResult
from checkllm.reporting.comparison import ComparisonReport
from checkllm.reporting.github import (
    _MARKER,
    generate_pr_comment,
    post_pr_comment,
)


def _sample_results() -> dict[str, list[CheckResult]]:
    return {
        "test_foo": [
            CheckResult(
                passed=True,
                score=0.9,
                reasoning="Good",
                cost=0.001,
                latency_ms=100,
                metric_name="hallucination",
            ),
            CheckResult(
                passed=False,
                score=0.3,
                reasoning="Bad output detected",
                cost=0.002,
                latency_ms=200,
                metric_name="relevance",
            ),
        ],
        "test_bar": [
            CheckResult(
                passed=True,
                score=1.0,
                reasoning="Perfect",
                cost=0.0,
                latency_ms=0,
                metric_name="contains",
            ),
        ],
    }


def _sample_comparison() -> ComparisonReport:
    return ComparisonReport(
        results_a={
            "test_foo": [
                CheckResult(
                    passed=True,
                    score=0.8,
                    reasoning="OK",
                    cost=0.001,
                    latency_ms=100,
                    metric_name="hallucination",
                ),
            ],
        },
        results_b={
            "test_foo": [
                CheckResult(
                    passed=True,
                    score=0.95,
                    reasoning="Great",
                    cost=0.002,
                    latency_ms=150,
                    metric_name="hallucination",
                ),
            ],
        },
        label_a="Baseline",
        label_b="Current",
    )


class TestGeneratePrComment:
    def test_contains_marker(self):
        comment = generate_pr_comment(_sample_results())
        assert _MARKER in comment

    def test_contains_report_heading(self):
        comment = generate_pr_comment(_sample_results())
        assert "checkllm Report" in comment

    def test_contains_pass_fail_counts(self):
        comment = generate_pr_comment(_sample_results())
        assert "2/3" in comment  # 2 passed out of 3

    def test_contains_collapsible_sections(self):
        comment = generate_pr_comment(_sample_results())
        assert "<details>" in comment
        assert "<summary>" in comment
        assert "</details>" in comment

    def test_contains_test_names(self):
        comment = generate_pr_comment(_sample_results())
        assert "test_foo" in comment
        assert "test_bar" in comment

    def test_contains_metric_names(self):
        comment = generate_pr_comment(_sample_results())
        assert "hallucination" in comment
        assert "relevance" in comment
        assert "contains" in comment

    def test_contains_scores(self):
        comment = generate_pr_comment(_sample_results())
        assert "0.90" in comment
        assert "0.30" in comment
        assert "1.00" in comment

    def test_contains_cost(self):
        comment = generate_pr_comment(_sample_results())
        assert "$0.003" in comment or "$0.0030" in comment

    def test_without_comparison(self):
        comment = generate_pr_comment(_sample_results())
        assert "Comparison" not in comment

    def test_with_comparison(self):
        comment = generate_pr_comment(_sample_results(), comparison=_sample_comparison())
        assert "Comparison" in comment
        assert "Baseline" in comment
        assert "Current" in comment
        assert "+0.150" in comment  # 0.95 - 0.80

    def test_comparison_summary_stats(self):
        comment = generate_pr_comment(_sample_results(), comparison=_sample_comparison())
        assert "Pass rate" in comment
        assert "Avg score" in comment
        assert "Total cost" in comment

    def test_empty_results(self):
        comment = generate_pr_comment({})
        assert _MARKER in comment
        assert "0/0" in comment


class TestPostPrComment:
    def test_raises_without_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token"):
                post_pr_comment("hello", "owner/repo", 1)

    def test_raises_if_httpx_missing(self, monkeypatch):
        """Simulates httpx not being installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="httpx"):
            post_pr_comment("hello", "owner/repo", 1, token="fake-token")

    def test_creates_new_comment(self):
        """Mock HTTP calls to verify a new comment is created."""
        mock_httpx = MagicMock()
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = []
        mock_get_response.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_post_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            post_pr_comment("test body", "owner/repo", 42, token="fake-token")

            mock_httpx.get.assert_called_once()
            mock_httpx.post.assert_called_once()
            # Verify the POST was to the correct URL
            call_args = mock_httpx.post.call_args
            assert "owner/repo" in call_args[0][0]
            assert call_args[1]["json"]["body"] == "test body"

    def test_updates_existing_comment(self):
        """Mock HTTP calls to verify an existing comment is updated."""
        mock_httpx = MagicMock()
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = [
            {"id": 999, "body": f"old content {_MARKER}"},
        ]
        mock_get_response.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_get_response

        mock_patch_response = MagicMock()
        mock_patch_response.raise_for_status = MagicMock()
        mock_httpx.patch.return_value = mock_patch_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            post_pr_comment("updated body", "owner/repo", 42, token="fake-token")

            mock_httpx.get.assert_called_once()
            mock_httpx.patch.assert_called_once()
            # Should NOT have called post
            mock_httpx.post.assert_not_called()
            # Verify PATCH to the correct comment
            call_args = mock_httpx.patch.call_args
            assert "999" in call_args[0][0]
            assert call_args[1]["json"]["body"] == "updated body"

    def test_uses_env_token(self, monkeypatch):
        """Falls back to GITHUB_TOKEN env variable."""
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")

        mock_httpx = MagicMock()
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = []
        mock_get_response.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_post_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            post_pr_comment("body", "owner/repo", 1)

            # Verify the token was used in headers
            call_args = mock_httpx.get.call_args
            headers = call_args[1]["headers"]
            assert "env-token" in headers["Authorization"]
