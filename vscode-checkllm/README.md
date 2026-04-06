# checkllm for VS Code

Inline LLM test results for [checkllm](https://pypi.org/project/checkllm/).

## Features

- **CodeLens annotations** above test functions showing pass/fail count, average score, and cost
- **Gutter indicators** with green/red dots next to each test
- **Hover details** showing a table of per-check scores and status
- **Auto-refresh** when `.checkllm/` result files change
- **Run checkllm** and **Estimate costs** directly from the command palette
- **Status bar** showing total pass/fail count

## Requirements

- [checkllm](https://pypi.org/project/checkllm/) Python package installed
- A `.checkllm/` directory with snapshot results (created by running `checkllm run tests/ --snapshot`)

## Quick Start

1. Install the extension
2. Open a project with checkllm tests
3. Run your tests: `checkllm run tests/ --snapshot .checkllm/snapshots/latest.json`
4. Open a test file — results appear inline

## Commands

| Command | Description |
|---------|-------------|
| `checkllm: Run checkllm Tests` | Run tests in a terminal |
| `checkllm: Estimate checkllm Costs` | Estimate costs before running |
| `checkllm: Refresh checkllm Results` | Reload results from disk |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `checkllm.judgeBackend` | `auto` | Judge backend (auto, openai, anthropic, gemini, ollama, litellm) |
| `checkllm.judgeModel` | `gpt-4o` | Judge model name |
| `checkllm.defaultThreshold` | `0.8` | Pass/fail threshold |
| `checkllm.budget` | `5.0` | Max USD per run |
| `checkllm.autoRefresh` | `true` | Auto-refresh when results change |

## Development

```bash
cd vscode-checkllm
npm install
npm run compile
```

To test: press F5 in VS Code to launch an Extension Development Host.

## Building

```bash
npm install -g @vscode/vsce
cd vscode-checkllm
vsce package
```

This produces `checkllm-0.1.0.vsix` which can be installed locally or published to the marketplace.
