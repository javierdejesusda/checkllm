/**
 * checkllm VS Code extension entry point.
 *
 * Registers three commands (runTest, runSuite, dashboard), a tree
 * view of discovered `.checkllm.yaml` files, and a status bar item
 * that surfaces the last run's pass/fail count. Heavy lifting is
 * delegated to the checkllm CLI via an integrated terminal.
 */

import * as path from 'path';
import * as vscode from 'vscode';

import {CheckllmTreeDataProvider} from './treeView';
import {StatusBar} from './statusBar';

const CHECKLLM_TERMINAL_NAME = 'checkllm';

function getOrCreateTerminal(): vscode.Terminal {
  const existing = vscode.window.terminals.find(
      (t) => t.name === CHECKLLM_TERMINAL_NAME);
  if (existing) {
    return existing;
  }
  return vscode.window.createTerminal({name: CHECKLLM_TERMINAL_NAME});
}

function runInTerminal(command: string): void {
  const terminal = getOrCreateTerminal();
  terminal.show(true);
  terminal.sendText(command, true);
}

async function runCurrentTest(statusBar: StatusBar): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage(
        'checkllm: open a .checkllm.yaml file before running.');
    return;
  }
  const doc = editor.document;
  if (!/\.checkllm\.ya?ml$/.test(doc.fileName)) {
    vscode.window.showWarningMessage(
        'checkllm: the current file is not a .checkllm.yaml test file.');
    return;
  }
  const filePath = doc.uri.fsPath;
  runInTerminal(`checkllm run "${filePath}"`);
  statusBar.markRunning(path.basename(filePath));
}

function runSuite(statusBar: StatusBar): void {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    vscode.window.showWarningMessage(
        'checkllm: open a workspace folder to run the suite.');
    return;
  }
  runInTerminal('checkllm run tests/');
  statusBar.markRunning('suite');
}

function openDashboard(): void {
  runInTerminal('checkllm dashboard');
}

export function activate(context: vscode.ExtensionContext): void {
  const statusBar = new StatusBar();
  context.subscriptions.push(statusBar);

  const tree = new CheckllmTreeDataProvider();
  context.subscriptions.push(vscode.window.registerTreeDataProvider(
      'checkllmTests', tree));

  context.subscriptions.push(
      vscode.commands.registerCommand(
          'checkllm.runTest', () => runCurrentTest(statusBar)),
      vscode.commands.registerCommand(
          'checkllm.runSuite', () => runSuite(statusBar)),
      vscode.commands.registerCommand('checkllm.dashboard', openDashboard),
      vscode.commands.registerCommand(
          'checkllm.refreshTree', () => tree.refresh()),
  );

  // Watch for result snapshots and update the status bar.
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (folder) {
    const pattern = new vscode.RelativePattern(
        folder, '.checkllm/ci_snapshot.json');
    const watcher = vscode.workspace.createFileSystemWatcher(pattern);
    const refresh = (uri: vscode.Uri) => statusBar.updateFromSnapshot(uri);
    watcher.onDidChange(refresh);
    watcher.onDidCreate(refresh);
    context.subscriptions.push(watcher);
    statusBar.loadSnapshotFromWorkspace(folder.uri);
  }
}

export function deactivate(): void {
  // Nothing to clean up — subscriptions handle disposal.
}
