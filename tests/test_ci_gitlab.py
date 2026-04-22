"""Tests for the GitLab CI helpers."""

from __future__ import annotations

from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from checkllm.ci import gitlab
from checkllm.cli import app


runner = CliRunner()


class TestDetect:
    """Environment-based detection of a GitLab CI job."""

    def test_detect_true_when_env_present(self):
        env = {"GITLAB_CI": "true", "CI_PROJECT_ID": "42"}
        with patch.dict("os.environ", env, clear=True):
            assert gitlab.detect() is True

    def test_detect_false_without_gitlab_ci(self):
        env = {"CI_PROJECT_ID": "42"}
        with patch.dict("os.environ", env, clear=True):
            assert gitlab.detect() is False

    def test_detect_false_without_project_id(self):
        env = {"GITLAB_CI": "true"}
        with patch.dict("os.environ", env, clear=True):
            assert gitlab.detect() is False


class TestContextFromEnv:
    """Parsing the GitLab environment into a GitLabContext."""

    def test_context_from_env_populates_fields(self):
        env = {
            "GITLAB_CI": "true",
            "CI_PROJECT_ID": "99",
            "CI_MERGE_REQUEST_IID": "7",
            "CI_SERVER_URL": "https://gitlab.example.com/",
            "GITLAB_TOKEN": "glpat-xxx",
        }
        with patch.dict("os.environ", env, clear=True):
            ctx = gitlab.context_from_env()
        assert ctx is not None
        assert ctx.project_id == "99"
        assert ctx.mr_iid == "7"
        assert ctx.server_url == "https://gitlab.example.com"
        assert ctx.token == "glpat-xxx"
        assert ctx.token_is_job_token is False

    def test_context_falls_back_to_job_token(self):
        env = {
            "GITLAB_CI": "true",
            "CI_PROJECT_ID": "99",
            "CI_JOB_TOKEN": "job-abc",
        }
        with patch.dict("os.environ", env, clear=True):
            ctx = gitlab.context_from_env()
        assert ctx is not None
        assert ctx.token == "job-abc"
        assert ctx.token_is_job_token is True

    def test_context_returns_none_outside_gitlab(self):
        with patch.dict("os.environ", {}, clear=True):
            assert gitlab.context_from_env() is None


class TestPostMrComment:
    """MR comment posting respects the parsed context."""

    def test_skip_when_no_context(self):
        with patch.dict("os.environ", {}, clear=True):
            assert gitlab.post_mr_comment("body") is False

    def test_skip_when_no_mr(self):
        env = {
            "GITLAB_CI": "true",
            "CI_PROJECT_ID": "1",
            "GITLAB_TOKEN": "t",
        }
        with patch.dict("os.environ", env, clear=True):
            assert gitlab.post_mr_comment("body") is False

    def test_posts_using_private_token(self):
        env = {
            "GITLAB_CI": "true",
            "CI_PROJECT_ID": "1",
            "CI_MERGE_REQUEST_IID": "2",
            "CI_SERVER_URL": "https://gitlab.com",
            "GITLAB_TOKEN": "glpat-token",
        }

        captured: dict = {}

        class _FakeResp:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_urlopen(req, timeout=10.0):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["data"] = req.data
            return _FakeResp()

        with patch.dict("os.environ", env, clear=True), \
             patch("checkllm.ci.gitlab.urlopen", _fake_urlopen):
            result = gitlab.post_mr_comment("hello")

        assert result is True
        assert "merge_requests/2/notes" in captured["url"]
        assert captured["headers"].get("private-token") == "glpat-token"
        assert b"hello" in captured["data"]


class TestTemplate:
    """``gitlab_template`` produces a valid YAML snippet."""

    def test_template_is_valid_yaml(self):
        snippet = gitlab.gitlab_template()
        parsed = yaml.safe_load(snippet)
        assert "checkllm" in parsed
        assert "script" in parsed["checkllm"]

    def test_template_respects_budget(self):
        snippet = gitlab.gitlab_template(budget=3.5)
        assert "--budget 3.50" in snippet

    def test_template_respects_python_version(self):
        snippet = gitlab.gitlab_template(python_version="3.12")
        assert "python:3.12-slim" in snippet


class TestFormatMrComment:
    """Summary generator for MR comments."""

    def test_format_includes_totals(self):
        class _R:
            def __init__(self, passed):
                self.passed = passed

        results = {"test_a": [_R(True), _R(False)], "test_b": [_R(True)]}
        body = gitlab.format_mr_comment(results)
        assert "Tests: 2" in body.replace("**", "")
        assert "Checks: 3" in body.replace("**", "")
        assert "Passed: 2" in body.replace("**", "")
        assert "Failed: 1" in body.replace("**", "")


class TestCliCommand:
    """The ``checkllm ci-gitlab-template`` command prints valid YAML."""

    def test_cli_prints_yaml(self):
        result = runner.invoke(app, ["ci-gitlab-template"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert isinstance(parsed, dict)
        assert "checkllm" in parsed

    def test_cli_writes_to_output(self, tmp_path):
        out = tmp_path / "gitlab-ci.yml"
        result = runner.invoke(
            app, ["ci-gitlab-template", "--output", str(out), "--budget", "2.5"]
        )
        assert result.exit_code == 0
        content = out.read_text()
        assert "--budget 2.50" in content


class TestCiAutoDetectsGitlab:
    """The ``checkllm ci`` command prints a GitLab banner when detected."""

    def test_ci_detects_gitlab(self, tmp_path):
        test_file = tmp_path / "test_ex.py"
        test_file.write_text(
            "def test_basic(check):\n"
            "    check.contains('hello world', 'hello')\n"
        )
        env = {
            "GITLAB_CI": "true",
            "CI_PROJECT_ID": "5",
            "CI_MERGE_REQUEST_IID": "11",
            "CI_SERVER_URL": "https://gitlab.example.com",
        }
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["ci", str(test_file), "--no-comment"])
        assert "GitLab CI" in result.output
        assert "MR: !11" in result.output
