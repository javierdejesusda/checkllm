"""Tests for the CI/CD integration template generators."""

from __future__ import annotations

import os
import tempfile

import yaml

from checkllm.cicd.github_action import GitHubActionGenerator
from checkllm.cicd.gitlab_ci import GitLabCIGenerator


class TestGitHubActionGenerator:
    """Tests for GitHub Actions workflow generation."""

    def test_generate_returns_string(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_valid_yaml(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "jobs" in parsed

    def test_default_triggers(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        parsed = yaml.safe_load(result)
        # PyYAML parses bare `on:` as boolean True key
        on = parsed.get("on") or parsed.get(True)
        assert on is not None
        assert "push" in on
        assert "pull_request" in on

    def test_schedule_included_when_set(self):
        gen = GitHubActionGenerator()
        result = gen.generate(schedule="0 3 * * *")
        parsed = yaml.safe_load(result)
        on = parsed.get("on") or parsed.get(True)
        assert "schedule" in on
        assert on["schedule"][0]["cron"] == "0 3 * * *"

    def test_schedule_omitted_when_none(self):
        gen = GitHubActionGenerator()
        result = gen.generate(schedule=None)
        parsed = yaml.safe_load(result)
        on = parsed.get("on") or parsed.get(True)
        assert "schedule" not in on

    def test_budget_included_in_command(self):
        gen = GitHubActionGenerator()
        result = gen.generate(budget=5.00)
        assert "--budget 5.00" in result

    def test_budget_omitted_when_none(self):
        gen = GitHubActionGenerator()
        result = gen.generate(budget=None)
        assert "--budget" not in result

    def test_python_version(self):
        gen = GitHubActionGenerator()
        result = gen.generate(python_version="3.12")
        assert "3.12" in result

    def test_pr_comment_step_present(self):
        gen = GitHubActionGenerator()
        result = gen.generate(post_pr_comment=True)
        assert "Post PR comment" in result
        assert "actions/github-script" in result

    def test_pr_comment_step_absent(self):
        gen = GitHubActionGenerator()
        result = gen.generate(post_pr_comment=False)
        assert "Post PR comment" not in result

    def test_artifact_upload(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        assert "upload-artifact" in result
        assert "eval-results.json" in result

    def test_custom_eval_command(self):
        gen = GitHubActionGenerator()
        result = gen.generate(eval_command="python -m checkllm run")
        assert "python -m checkllm run" in result

    def test_checkout_step(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        assert "actions/checkout@v4" in result

    def test_pip_cache(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        assert "cache: pip" in result

    def test_secrets_referenced(self):
        gen = GitHubActionGenerator()
        result = gen.generate()
        assert "OPENAI_API_KEY" in result
        assert "ANTHROPIC_API_KEY" in result

    def test_generate_pr_comment_script(self):
        gen = GitHubActionGenerator()
        script = gen.generate_pr_comment_script()
        assert isinstance(script, str)
        assert "GITHUB_TOKEN" in script
        assert "eval-results.json" in script
        assert "def main" in script

    def test_save_creates_files(self):
        gen = GitHubActionGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, ".github", "workflows")
            paths = gen.save(output_dir=output_dir)
            assert len(paths) == 2
            for p in paths:
                assert os.path.exists(p)


class TestGitLabCIGenerator:
    """Tests for GitLab CI pipeline generation."""

    def test_generate_returns_string(self):
        gen = GitLabCIGenerator()
        result = gen.generate()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_valid_yaml(self):
        gen = GitLabCIGenerator()
        result = gen.generate()
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "stages" in parsed

    def test_stages_defined(self):
        gen = GitLabCIGenerator()
        result = gen.generate()
        parsed = yaml.safe_load(result)
        assert "test" in parsed["stages"]
        assert "report" in parsed["stages"]

    def test_budget_included_in_command(self):
        gen = GitLabCIGenerator()
        result = gen.generate(budget=10.00)
        assert "--budget 10.00" in result

    def test_budget_omitted_when_none(self):
        gen = GitLabCIGenerator()
        result = gen.generate(budget=None)
        assert "--budget" not in result

    def test_python_version_in_image(self):
        gen = GitLabCIGenerator()
        result = gen.generate(python_version="3.12")
        assert "python:3.12-slim" in result

    def test_artifacts_configured(self):
        gen = GitLabCIGenerator()
        result = gen.generate()
        assert "eval-results.json" in result
        assert "artifacts" in result

    def test_fail_on_regression_true(self):
        gen = GitLabCIGenerator()
        result = gen.generate(fail_on_regression=True)
        assert "allow_failure: false" in result

    def test_fail_on_regression_false(self):
        gen = GitLabCIGenerator()
        result = gen.generate(fail_on_regression=False)
        assert "allow_failure: true" in result

    def test_cache_configured(self):
        gen = GitLabCIGenerator()
        result = gen.generate()
        assert "PIP_CACHE_DIR" in result
        assert ".pip-cache" in result

    def test_custom_eval_command(self):
        gen = GitLabCIGenerator()
        result = gen.generate(eval_command="checkllm run --all")
        assert "checkllm run --all" in result
