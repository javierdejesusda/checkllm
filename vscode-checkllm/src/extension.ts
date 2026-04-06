import * as vscode from 'vscode';
import * as path from 'path';
import { ResultLoader } from './resultLoader';
import { CheckllmCodeLensProvider } from './codeLensProvider';
import { DecorationProvider } from './decorationProvider';
import { CheckllmHoverProvider } from './hoverProvider';
import { runCheckllm, estimateCheckllm, showDetails } from './commands';

let loader: ResultLoader;
let codeLensProvider: CheckllmCodeLensProvider;
let decorationProvider: DecorationProvider;
let fileWatcher: vscode.FileSystemWatcher | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceRoot) return;

  // Initialize result loader
  loader = new ResultLoader(workspaceRoot);
  loader.loadResults();

  // CodeLens provider
  codeLensProvider = new CheckllmCodeLensProvider(loader);
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      { language: 'python', scheme: 'file' },
      codeLensProvider
    )
  );

  // Hover provider
  const hoverProvider = new CheckllmHoverProvider(loader);
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      { language: 'python', scheme: 'file' },
      hoverProvider
    )
  );

  // Decoration provider
  decorationProvider = new DecorationProvider(loader);

  // Update decorations on editor change
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(editor => {
      if (editor) decorationProvider.updateDecorations(editor);
    })
  );

  // Update decorations for current editor
  if (vscode.window.activeTextEditor) {
    decorationProvider.updateDecorations(vscode.window.activeTextEditor);
  }

  // File watcher for .checkllm/ directory
  const config = vscode.workspace.getConfiguration('checkllm');
  if (config.get<boolean>('autoRefresh', true)) {
    const pattern = new vscode.RelativePattern(
      workspaceRoot,
      '.checkllm/**/*.{json,db}'
    );
    fileWatcher = vscode.workspace.createFileSystemWatcher(pattern);

    const refresh = () => {
      loader.loadResults();
      codeLensProvider.refresh();
      if (vscode.window.activeTextEditor) {
        decorationProvider.updateDecorations(vscode.window.activeTextEditor);
      }
    };

    fileWatcher.onDidChange(refresh);
    fileWatcher.onDidCreate(refresh);
    fileWatcher.onDidDelete(refresh);
    context.subscriptions.push(fileWatcher);
  }

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('checkllm.run', runCheckllm),
    vscode.commands.registerCommand('checkllm.estimate', estimateCheckllm),
    vscode.commands.registerCommand('checkllm.refresh', () => {
      loader.loadResults();
      codeLensProvider.refresh();
      if (vscode.window.activeTextEditor) {
        decorationProvider.updateDecorations(vscode.window.activeTextEditor);
      }
      vscode.window.showInformationMessage('checkllm: Results refreshed');
    }),
    vscode.commands.registerCommand('checkllm.showDetails', showDetails)
  );

  // Status bar
  const statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left, 100
  );
  const allResults = loader.getAllResults();
  const totalChecks = allResults.reduce((sum, r) => sum + r.checks.length, 0);
  const passedChecks = allResults.reduce(
    (sum, r) => sum + r.checks.filter(c => c.passed).length, 0
  );
  statusItem.text = `$(beaker) checkllm: ${passedChecks}/${totalChecks}`;
  statusItem.tooltip = 'Click to refresh checkllm results';
  statusItem.command = 'checkllm.refresh';
  statusItem.show();
  context.subscriptions.push(statusItem);
}

export function deactivate(): void {
  fileWatcher?.dispose();
}
