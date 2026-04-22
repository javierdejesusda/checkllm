# checkllm for VS Code

Run [checkllm](https://github.com/checkllm/checkllm) evaluations without
leaving your editor.

## Features

- **Run Current Test** — executes `checkllm run` on the `.checkllm.yaml`
  file in the active editor.
- **Run Test Suite** — executes `checkllm run tests/` in an integrated
  terminal.
- **Open Dashboard** — launches `checkllm dashboard` in a terminal.
- **Tree view** — the checkllm activity bar lists every
  `*.checkllm.yaml` file in the current workspace.
- **Status bar** — the last run's pass/fail totals are kept in sync by
  watching `.checkllm/ci_snapshot.json`.
- **Syntax highlighting** — `*.checkllm.yaml` files are highlighted
  with a custom grammar that colours metric names and the top-level
  test schema keys.

## Requirements

- checkllm installed on `PATH` (`pip install checkllm`).
- VS Code `^1.85.0`.

## Development

```bash
cd editor/vscode
npm install
npm run compile
```

Press `F5` in VS Code with this folder open to launch the Extension
Development Host.

## Publishing

The extension is published through the VS Code Marketplace via
[`vsce`](https://github.com/microsoft/vscode-vsce).

```bash
npm install -g @vscode/vsce
cd editor/vscode
npm run compile
vsce package              # produces checkllm-vscode-<version>.vsix
vsce publish              # requires a personal access token
```

Alternatively, distribute the generated `.vsix` manually via *Extensions
› Install from VSIX* in VS Code.

## Commands

| Command id              | Default title        |
| ----------------------- | -------------------- |
| `checkllm.runTest`      | Run Current Test     |
| `checkllm.runSuite`     | Run Test Suite       |
| `checkllm.dashboard`    | Open Dashboard       |
| `checkllm.refreshTree`  | Refresh Test Tree    |

## License

Apache-2.0
