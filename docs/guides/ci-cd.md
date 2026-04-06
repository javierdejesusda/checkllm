# CI/CD Integration

## GitHub Actions

### Quick Setup

```bash
checkllm init --ci
```

This generates `.github/workflows/checkllm.yml`:

```yaml
name: checkllm
on:
  pull_request:
    branches: [main]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install checkllm[all] pytest
      - run: checkllm ci --budget 5.0
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### The `checkllm ci` Command

```bash
checkllm ci                           # Run tests, auto-comment on PR
checkllm ci --fail-on-regression      # Exit 1 if scores drop
checkllm ci --compare main            # Compare against main baseline
checkllm ci --no-comment              # Skip PR comment
checkllm ci --budget 5.0              # Cap spend
```

When running in GitHub Actions, `checkllm ci` automatically:

1. Detects the PR number from `GITHUB_EVENT_PATH`
2. Runs tests with snapshots
3. Posts a formatted comment on the PR with results
4. Updates the comment on subsequent pushes
