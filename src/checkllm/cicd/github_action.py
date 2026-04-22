"""GitHub Actions workflow generator for checkllm evaluations.

Generates ready-to-use workflow YAML that runs LLM evaluations on pull
requests, pushes to main, and optional nightly schedules.

Usage::

    from checkllm.cicd.github_action import GitHubActionGenerator

    gen = GitHubActionGenerator()
    yaml_str = gen.generate(budget=5.00, schedule="0 3 * * *")
    gen.save()
"""

from __future__ import annotations

import os
import textwrap


class GitHubActionGenerator:
    """Generates GitHub Actions workflow YAML for checkllm evaluations."""

    def generate(
        self,
        eval_command: str = "pytest tests/ -m checkllm",
        python_version: str = "3.11",
        budget: float | None = None,
        fail_on_regression: bool = True,
        post_pr_comment: bool = True,
        schedule: str | None = None,
    ) -> str:
        """Generate a GitHub Actions workflow YAML string.

        Args:
            eval_command: Shell command that runs the evaluation suite.
            python_version: Python version to install in the runner.
            budget: Optional maximum USD spend for the evaluation run.
                When set, ``--budget <value>`` is appended to the command.
            fail_on_regression: If True the step fails when a regression
                is detected.
            post_pr_comment: If True a step posts evaluation results as
                a PR comment.
            schedule: Optional cron expression for nightly runs
                (e.g. ``"0 3 * * *"``).

        Returns:
            A complete GitHub Actions workflow YAML string.
        """
        eval_cmd = eval_command
        if budget is not None:
            eval_cmd = f"{eval_command} --budget {budget:.2f}"

        lines: list[str] = []
        lines.append("name: checkllm Evaluation")
        lines.append("")

        lines.append("on:")
        lines.append("  push:")
        lines.append("    branches: [main]")
        lines.append("  pull_request:")
        lines.append("    branches: [main]")
        if schedule:
            lines.append("  schedule:")
            lines.append(f'    - cron: "{schedule}"')
        lines.append("")

        lines.append("permissions:")
        lines.append("  contents: read")
        lines.append("  pull-requests: write")
        lines.append("")

        lines.append("jobs:")
        lines.append("  evaluate:")
        lines.append("    runs-on: ubuntu-latest")
        lines.append("    steps:")
        lines.append("      - name: Checkout repository")
        lines.append("        uses: actions/checkout@v4")
        lines.append("")
        lines.append(f"      - name: Set up Python {python_version}")
        lines.append("        uses: actions/setup-python@v5")
        lines.append("        with:")
        lines.append(f'          python-version: "{python_version}"')
        lines.append("          cache: pip")
        lines.append("")
        lines.append("      - name: Install dependencies")
        lines.append("        run: |")
        lines.append("          python -m pip install --upgrade pip")
        lines.append("          pip install checkllm")
        lines.append(
            "          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"
        )
        lines.append("")
        lines.append("      - name: Run checkllm evaluations")
        lines.append("        env:")
        lines.append("          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}")
        lines.append("          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}")
        if fail_on_regression:
            lines.append("        continue-on-error: false")
        lines.append("        run: |")
        lines.append(f"          {eval_cmd} --json-report eval-results.json || export EVAL_EXIT=$?")
        lines.append('          echo "EVAL_EXIT=${EVAL_EXIT:-0}" >> $GITHUB_ENV')
        lines.append("")
        lines.append("      - name: Upload evaluation results")
        lines.append("        if: always()")
        lines.append("        uses: actions/upload-artifact@v4")
        lines.append("        with:")
        lines.append("          name: checkllm-eval-results")
        lines.append("          path: eval-results.json")
        lines.append("          retention-days: 30")

        if post_pr_comment:
            lines.append("")
            lines.append("      - name: Post PR comment")
            lines.append("        if: github.event_name == 'pull_request'")
            lines.append("        uses: actions/github-script@v7")
            lines.append("        with:")
            lines.append("          script: |")
            lines.append("            const fs = require('fs');")
            lines.append("            let body = '## checkllm Evaluation Results\\n\\n';")
            lines.append("            try {")
            lines.append(
                "              const report = fs.readFileSync('eval-results.json', 'utf8');"
            )
            lines.append("              const data = JSON.parse(report);")
            lines.append("              body += `**Pass rate:** ${data.pass_rate || 'N/A'}\\n`;")
            lines.append("              body += `**Total checks:** ${data.total || 'N/A'}\\n`;")
            lines.append("              body += `**Cost:** $${data.total_cost || '0.00'}\\n`;")
            lines.append("            } catch {")
            lines.append(
                "              body += '_No structured results found. Check the workflow logs._\\n';"
            )
            lines.append("            }")
            lines.append("            github.rest.issues.createComment({")
            lines.append("              issue_number: context.issue.number,")
            lines.append("              owner: context.repo.owner,")
            lines.append("              repo: context.repo.repo,")
            lines.append("              body: body,")
            lines.append("            });")

        lines.append("")
        lines.append("      - name: Check for regressions")
        lines.append("        if: env.EVAL_EXIT != '0'")
        lines.append("        run: |")
        lines.append('          echo "::error::checkllm evaluation detected regressions"')
        lines.append("          exit 1")
        lines.append("")

        return "\n".join(lines)

    def generate_pr_comment_script(self) -> str:
        """Generate a standalone Python script that posts eval results as PR comments.

        Returns:
            A Python script string that can be saved to a file and
            executed in a GitHub Actions step.
        """
        return textwrap.dedent("""\
            #!/usr/bin/env python3
            \"\"\"Post checkllm evaluation results as a GitHub PR comment.\"\"\"

            import json
            import os
            import sys
            from urllib.request import Request, urlopen

            def main() -> None:
                token = os.environ.get("GITHUB_TOKEN", "")
                repo = os.environ.get("GITHUB_REPOSITORY", "")
                event_path = os.environ.get("GITHUB_EVENT_PATH", "")

                if not all([token, repo, event_path]):
                    print("Missing environment variables. Skipping PR comment.")
                    sys.exit(0)

                with open(event_path, "r") as f:
                    event = json.load(f)

                pr_number = event.get("pull_request", {}).get("number")
                if not pr_number:
                    print("Not a pull request event. Skipping.")
                    sys.exit(0)

                body = "## checkllm Evaluation Results\\n\\n"
                try:
                    with open("eval-results.json", "r") as f:
                        data = json.load(f)
                    body += f"**Pass rate:** {data.get('pass_rate', 'N/A')}\\n"
                    body += f"**Total checks:** {data.get('total', 'N/A')}\\n"
                    body += f"**Failed:** {data.get('failed', 0)}\\n"
                    body += f"**Cost:** ${data.get('total_cost', '0.00')}\\n"
                except FileNotFoundError:
                    body += "_No results file found. Check workflow logs._\\n"

                url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
                payload = json.dumps({"body": body}).encode()
                req = Request(url, data=payload, method="POST")
                req.add_header("Authorization", f"Bearer {token}")
                req.add_header("Accept", "application/vnd.github+json")
                req.add_header("Content-Type", "application/json")

                with urlopen(req) as resp:
                    print(f"Posted PR comment (status {resp.status})")

            if __name__ == "__main__":
                main()
        """)

    def save(self, output_dir: str = ".github/workflows") -> list[str]:
        """Save generated files to disk.

        Args:
            output_dir: Directory where workflow files are written.

        Returns:
            List of file paths that were created.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []

        workflow_path = os.path.join(output_dir, "checkllm-eval.yml")
        with open(workflow_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate())
        paths.append(workflow_path)

        scripts_dir = os.path.join(output_dir, "..", "scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        script_path = os.path.join(scripts_dir, "post_eval_comment.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_pr_comment_script())
        paths.append(script_path)

        return paths
