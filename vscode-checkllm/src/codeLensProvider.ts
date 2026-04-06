import * as vscode from 'vscode';
import { ResultLoader } from './resultLoader';
import { TestResult } from './types';

/**
 * Shows CodeLens annotations above test functions with pass/fail status.
 *
 * Example display:
 *   checkllm: 3/3 passed (0.95 avg) | $0.0031
 *   def test_my_llm(check):
 */
export class CheckllmCodeLensProvider implements vscode.CodeLensProvider {
  private loader: ResultLoader;
  private onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.onDidChange.event;

  constructor(loader: ResultLoader) {
    this.loader = loader;
  }

  refresh(): void {
    this.onDidChange.fire();
  }

  provideCodeLenses(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken
  ): vscode.CodeLens[] {
    if (!document.fileName.endsWith('.py')) return [];

    const results = this.loader.getResultsForFile(document.fileName);
    if (results.length === 0) return [];

    const lenses: vscode.CodeLens[] = [];
    const text = document.getText();

    for (const result of results) {
      // Find the test function definition line
      const pattern = new RegExp(`^(\s*def\s+${this.escapeRegex(result.testName)}\s*\()`, 'm');
      const match = pattern.exec(text);
      if (!match) continue;

      const pos = document.positionAt(match.index);
      const range = new vscode.Range(pos, pos);

      const passed = result.checks.filter(c => c.passed).length;
      const total = result.checks.length;
      const avgScore = total > 0
        ? result.checks.reduce((sum, c) => sum + c.score, 0) / total
        : 0;

      const icon = result.allPassed ? '\u2705' : '\u274C';
      const title = `${icon} checkllm: ${passed}/${total} passed (${avgScore.toFixed(2)} avg) | $${result.totalCost.toFixed(4)}`;

      lenses.push(new vscode.CodeLens(range, {
        title,
        command: 'checkllm.showDetails',
        arguments: [result],
        tooltip: `Click to see check details for ${result.testName}`,
      }));
    }

    return lenses;
  }

  private escapeRegex(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\]/g, '\$&');
  }
}
