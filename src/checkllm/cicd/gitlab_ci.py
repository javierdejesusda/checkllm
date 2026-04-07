"""GitLab CI pipeline generator for checkllm evaluations.

Usage::

    from checkllm.cicd.gitlab_ci import GitLabCIGenerator

    gen = GitLabCIGenerator()
    yaml_str = gen.generate(budget=5.00)
"""

from __future__ import annotations

import textwrap


class GitLabCIGenerator:
    """Generates GitLab CI pipeline YAML for checkllm evaluations."""

    def generate(
        self,
        eval_command: str = "pytest tests/ -m checkllm",
        python_version: str = "3.11",
        budget: float | None = None,
        fail_on_regression: bool = True,
    ) -> str:
        """Generate a GitLab CI pipeline YAML string.

        Args:
            eval_command: Shell command that runs the evaluation suite.
            python_version: Python version for the Docker image tag.
            budget: Optional maximum USD spend for the evaluation run.
            fail_on_regression: When True the job fails if a regression
                is detected.

        Returns:
            A complete ``.gitlab-ci.yml`` string.
        """
        eval_cmd = eval_command
        if budget is not None:
            eval_cmd = f"{eval_command} --budget {budget:.2f}"

        allow_failure = "false" if fail_on_regression else "true"

        yaml = textwrap.dedent(f"""\
            stages:
              - test
              - report

            variables:
              PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

            .python-setup: &python-setup
              image: python:{python_version}-slim
              cache:
                key: pip-cache
                paths:
                  - .pip-cache/
              before_script:
                - python -m pip install --upgrade pip
                - pip install checkllm
                - if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

            checkllm-evaluation:
              <<: *python-setup
              stage: test
              script:
                - {eval_cmd} --json-report eval-results.json
              artifacts:
                when: always
                paths:
                  - eval-results.json
                expire_in: 30 days
                reports:
                  junit: eval-results.json
              allow_failure: {allow_failure}
              rules:
                - if: $CI_PIPELINE_SOURCE == "merge_request_event"
                - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

            checkllm-report:
              <<: *python-setup
              stage: report
              script:
                - |
                  python -c "
                  import json, sys
                  try:
                      with open('eval-results.json') as f:
                          data = json.load(f)
                      print('=== checkllm Evaluation Report ===')
                      print(f'Pass rate: {{data.get(\"pass_rate\", \"N/A\")}}')
                      print(f'Total checks: {{data.get(\"total\", \"N/A\")}}')
                      print(f'Cost: ${{data.get(\"total_cost\", \"0.00\")}}')
                  except FileNotFoundError:
                      print('No results file found.')
                      sys.exit(1)
                  "
              dependencies:
                - checkllm-evaluation
              rules:
                - if: $CI_PIPELINE_SOURCE == "merge_request_event"
                - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
        """)
        return yaml
