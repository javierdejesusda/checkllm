/**
 * Tree data provider that lists discovered checkllm test files.
 *
 * Scans the active workspace for ``*.checkllm.yaml`` files and
 * renders each one as a leaf in the sidebar. Clicking a leaf opens
 * the file in the active editor.
 */

import * as path from 'path';
import * as vscode from 'vscode';

export class CheckllmTreeDataProvider implements
    vscode.TreeDataProvider<vscode.Uri> {
  private readonly onDidChange =
      new vscode.EventEmitter<vscode.Uri|undefined|void>();
  readonly onDidChangeTreeData = this.onDidChange.event;

  refresh(): void {
    this.onDidChange.fire();
  }

  getTreeItem(element: vscode.Uri): vscode.TreeItem {
    const label = path.basename(element.fsPath);
    const item = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.None);
    item.resourceUri = element;
    item.tooltip = element.fsPath;
    item.command = {
      command: 'vscode.open',
      title: 'Open',
      arguments: [element],
    };
    item.iconPath = new vscode.ThemeIcon('beaker');
    return item;
  }

  async getChildren(element?: vscode.Uri): Promise<vscode.Uri[]> {
    if (element) {
      return [];
    }
    const matches = await vscode.workspace.findFiles(
        '**/*.checkllm.{yaml,yml}', '**/node_modules/**');
    matches.sort((a, b) => a.fsPath.localeCompare(b.fsPath));
    return matches;
  }
}
