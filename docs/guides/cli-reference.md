# CLI Reference

## Installation

To use the checkllm CLI, install the package via pip:

```bash
pip install checkllm
```

Or for development with all optional dependencies:

```bash
pip install checkllm[all]
```

## Core Commands

### run

Run LLM tests with rich terminal output.

```bash
# Basic test run
checkllm run tests/

# Run with snapshot comparison and fail on regression
checkllm run tests/ --compare baseline.json --snapshot current.json --fail-on-regression

# Generate HTML report
checkllm run tests/ --html-report report.html

# Run with JUnit XML output
checkllm run tests/ --junit-xml results.xml

# Estimate costs without running (dry run)
checkllm run tests/ --dry-run

# Run with budget limit
checkllm run tests/ --budget 5.0

# Run with a label for history tracking
checkllm run tests/ --label "v1.0.0-rc1"

# Disable caching for fresh results
checkllm run tests/ --no-cache
```

**Key Options:**
- `--compare PATH` — Path to baseline snapshot for regression comparison
- `--fail-on-regression` — Exit with code 1 if regression detected
- `--junit-xml PATH` — Write JUnit XML to this path
- `--html-report PATH` — Generate HTML report to this path
- `--snapshot PATH` — Save snapshot to this path for future comparison
- `--budget FLOAT` — Maximum USD to spend on judge calls
- `--no-cache` — Disable judge response caching
- `--label TEXT` — Label for this run in history
- `--dry-run` — Estimate costs without running tests

### snapshot

Run tests and save results as a regression baseline snapshot.

```bash
# Create a baseline snapshot
checkllm snapshot tests/

# Save snapshot to custom location
checkllm snapshot tests/ --output snapshots/baseline_v1.json

# Short option
checkllm snapshot tests/ -o baseline.json
```

**Key Options:**
- `--output PATH` / `-o` — Snapshot output path (default: auto-generated in config snapshot_dir)

### report

Run tests and generate an HTML report.

```bash
# Generate HTML report
checkllm report tests/

# Custom report name
checkllm report tests/ --output evaluation_report.html

# Generate both HTML and JUnit XML
checkllm report tests/ --output report.html --junit-xml results.xml
```

**Key Options:**
- `--output PATH` / `-o` — Output file path (default: `checkllm_report.html`)
- `--junit-xml PATH` — Also write JUnit XML to this path

### eval

Evaluate a prompt template against a dataset.

```bash
# Basic evaluation with hallucination check
checkllm eval --prompt "Q: {input}\nA:" --dataset cases.yaml --metric hallucination

# Evaluate with custom threshold
checkllm eval --prompt "Summarize: {input}" --dataset data.json --metric relevance --threshold 0.9

# Evaluate with specific model
checkllm eval --prompt "Generate: {input}" --dataset corpus.csv --model claude-3-opus --metric rubric

# Save results as snapshot
checkllm eval --prompt "Q: {input}" --dataset qa.yaml --output results.json

# With budget limit
checkllm eval --prompt "Evaluate: {input}" --dataset test.json --budget 2.0

# With label for history
checkllm eval --prompt "Check: {input}" --dataset data.yaml --label "eval_batch_1"
```

**Key Options:**
- `--prompt TEXT` / `-p` — Prompt template with `{input}` placeholder (required)
- `--dataset PATH` / `-d` — Path to dataset file: YAML, JSON, or CSV (required)
- `--model TEXT` / `-M` — Model to generate outputs (default: from config)
- `--metric TEXT` / `-m` — Metric to evaluate: hallucination, relevance, toxicity, rubric, fluency, coherence, sentiment, correctness (default: `rubric`)
- `--threshold FLOAT` / `-t` — Pass/fail threshold (default: `0.8`)
- `--output PATH` / `-o` — Save results as snapshot JSON
- `--budget FLOAT` — Maximum USD to spend on judge calls
- `--no-cache` — Disable judge response caching
- `--label TEXT` / `-l` — Label for this run in history

### eval-yaml

Run evaluation from a YAML configuration file. Supports promptfoo-style YAML configs with prompts, providers, test cases, and assertions.

```bash
# Run from YAML config
checkllm eval-yaml evaluation.yaml

# Override budget
checkllm eval-yaml tests/eval.yml --budget 2.0

# Dry run to validate config without executing
checkllm eval-yaml config.yaml --dry-run
```

**Key Options:**
- `--budget FLOAT` — Override budget (USD)
- `--dry-run` — Parse config and show plan without running

### yaml-run

Run evaluations defined in a YAML config file.

```bash
# Basic YAML evaluation run
checkllm yaml-run config.yaml

# Save results to JSON
checkllm yaml-run evaluation.yml --output results.json
```

**Key Options:**
- `--output PATH` / `-o` — Save results JSON

### diff

Compare two snapshots and detect regressions.

```bash
# Compare baseline and current
checkllm diff --baseline baseline.json --current current.json

# Fail if regressions detected
checkllm diff --baseline v1.0.json --current v1.1.json --fail-on-regression
```

**Key Options:**
- `--baseline PATH` / `-b` — Path to baseline snapshot (required)
- `--current PATH` / `-c` — Path to current snapshot (required)
- `--fail-on-regression` — Exit 1 if regression detected

### history

View historical run data and trends.

```bash
# Show recent runs
checkllm history

# Show more runs
checkllm history --limit 50

# Show details for specific run
checkllm history --run 5

# Compare two runs
checkllm history --compare 5,10

# Show metric trend over time
checkllm history --trend "test_qa::hallucination"
```

**Key Options:**
- `--limit INT` / `-n` — Number of runs to show (default: `20`)
- `--run INT` / `-r` — Show details for a specific run
- `--trend TEXT` — Show score trend for test::metric (e.g., `test_qa::hallucination`)
- `--compare TEXT` — Compare two runs: `ID1,ID2`

### cache

Manage the judge response cache.

```bash
# Show cache statistics
checkllm cache --stats

# Clear the entire cache
checkllm cache --clear
```

**Key Options:**
- `--clear` — Clear the entire cache
- `--stats` — Show cache statistics

### estimate

Estimate the cost of running checks before executing them.

```bash
# Estimate costs for test directory
checkllm estimate tests/

# Estimate with specific model
checkllm estimate tests/ --model claude-3-sonnet

# Cost comparison with cheaper model
checkllm estimate tests/ --model gpt-4
```

**Key Options:**
- `--model TEXT` / `-m` — Model to estimate costs for (default: `gpt-4o`)

### list-metrics

List all available metrics and checks.

```bash
# Show all metrics
checkllm list-metrics

# Show only installed/available metrics
checkllm list-metrics --installed
```

**Key Options:**
- `--installed` — Show only installed/available metrics

### init

Scaffold a new checkllm project with tailored test files.

```bash
# Initialize in current directory
checkllm init

# Initialize with RAG template
checkllm init . --use-case rag

# Initialize with chatbot template and CI workflow
checkllm init . --use-case chatbot --ci

# Initialize specific directory
checkllm init /path/to/project --use-case agent
```

**Key Options:**
- `--use-case TEXT` / `-u` — What you're building: `rag`, `chatbot`, `agent`, `general`
- `--ci` — Also generate GitHub Actions workflow

### watch

Watch for file changes and re-run tests automatically.

```bash
# Watch tests directory
checkllm watch tests/

# Watch with custom poll interval
checkllm watch tests/ --interval 0.5

# Watch multiple paths
checkllm watch tests/ --watch src/ --watch configs/

# Watch with file pattern filter
checkllm watch tests/ --pattern "test_*.py" --pattern "fixtures/*.yaml"

# Watch with budget
checkllm watch tests/ --budget 10.0

# Watch without cache
checkllm watch tests/ --no-cache

# Watch with config profile
checkllm watch tests/ --profile production
```

**Key Options:**
- `--watch PATH` / `-w` — Additional paths to watch (can be repeated)
- `--interval FLOAT` / `-i` — Poll interval in seconds (default: `1.0`)
- `--debounce FLOAT` — Debounce delay in seconds (default: `0.5`)
- `--pattern TEXT` / `-p` — File patterns to watch (can be repeated)
- `--budget FLOAT` — Budget per run
- `--no-cache` — Disable cache
- `--profile TEXT` — Config profile to use

### dashboard

Launch the interactive web dashboard.

```bash
# Start dashboard on default port
checkllm dashboard

# Custom port
checkllm dashboard --port 9000

# Bind to specific host
checkllm dashboard --host 0.0.0.0

# Don't open browser automatically
checkllm dashboard --no-browser

# Use custom experiments database
checkllm dashboard --db .checkllm/custom_exp.db
```

**Key Options:**
- `--port INT` / `-p` — Port to serve on (default: `8484`)
- `--host TEXT` — Host to bind to (default: `localhost`)
- `--no-browser` — Don't open browser
- `--db PATH` — Experiments database path

### redteam

Run automated red teaming against an LLM.

```bash
# Basic red team scan
checkllm redteam "You are a helpful assistant"

# Specify number of attacks per type
checkllm redteam "You are a helpful assistant" --attacks 5

# Test specific vulnerability types
checkllm redteam "You are a helpful assistant" --type prompt_injection --type jailbreak

# Save report
checkllm redteam "You are helpful" --output report.json
```

**Key Options:**
- `--attacks INT` / `-n` — Attacks per type (default: `3`)
- `--type TEXT` / `-t` — Vulnerability types to test (can be repeated)
- `--output PATH` / `-o` — Save report JSON

### experiments

View and compare experiment tracking data.

```bash
# List experiment runs
checkllm experiments --list

# Show limit of runs
checkllm experiments --limit 50

# Compare two experiment runs
checkllm experiments --compare "id1,id2"

# Get best run for experiment
checkllm experiments --best "experiment_name"

# Use custom database
checkllm experiments --db custom.db
```

**Key Options:**
- `--list` / `-l` — List experiment runs
- `--compare TEXT` — Compare runs: `ID1,ID2`
- `--best TEXT` — Best run for experiment
- `--limit INT` / `-n` — Number of runs (default: `20`)
- `--db PATH` — Database path

### ci

Run tests in CI and post results as a PR comment.

Auto-detects GitHub Actions environment (GITHUB_TOKEN, PR number). Falls back to normal test run when not in GitHub Actions.

```bash
# Basic CI run
checkllm ci

# Run specific test path
checkllm ci tests/integration/

# Compare against baseline with regression detection
checkllm ci --compare main --fail-on-regression

# With budget limit
checkllm ci --budget 10.0

# Skip PR comment posting
checkllm ci --no-comment

# With config profile
checkllm ci --profile production
```

**Key Options:**
- `--fail-on-regression` — Exit 1 if regression detected
- `--compare TEXT` — Branch to compare against (snapshot baseline)
- `--budget FLOAT` — Maximum USD to spend
- `--no-comment` — Skip posting PR comment
- `--profile TEXT` — Config profile to use

## Global Options

These options are available for all commands:

- `--version` — Show version and exit
- `--help` — Show help message and exit

## Environment Variables

The following environment variables can be used to configure checkllm CLI behavior:

| Variable | Purpose |
|----------|---------|
| `CHECKLLM_JUDGE_MODEL` | Default judge model for evaluations |
| `CHECKLLM_BUDGET` | Default budget in USD for test runs |
| `CHECKLLM_ENGINE` | Judge backend: `openai` or `anthropic` |
| `CHECKLLM_CACHE` | Enable/disable caching: `true` or `false` |
| `CHECKLLM_LOG_LEVEL` | Logging level: `debug`, `info`, `warning`, `error` |
| `OPENAI_API_KEY` | OpenAI API key for gpt-4o and other models |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models |
| `GOOGLE_API_KEY` | Google API key for Gemini models |

**Examples:**

```bash
# Set budget for all runs in session
export CHECKLLM_BUDGET=5.0
checkllm run tests/

# Use Claude for judging
export CHECKLLM_JUDGE_MODEL=claude-3-opus
export ANTHROPIC_API_KEY=sk-ant-...
checkllm eval --prompt "..." --dataset data.yaml

# Enable debug logging
export CHECKLLM_LOG_LEVEL=debug
checkllm run tests/
```
