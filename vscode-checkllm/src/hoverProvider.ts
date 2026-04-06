import * as vscode from 'vscode';
import { ResultLoader } from './resultLoader';

/**
 * Shows a hover popup with check details when hovering over a test function.
 */
export class CheckllmHoverProvider implements vscode.HoverProvider {
  private loader: ResultLoader;

  constructor(loader: ResultLoader) {
    this.loader = loader;
  }

  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken
  ): vscode.Hover | undefined {
    if (!document.fileName.endsWith('.py')) return undefined;

    const line = document.lineAt(position.line).text;
    const defMatch = line.match(/^\s*def\s+(\w+)\s*\(/);
    if (!defMatch) return undefined;

    const funcName = defMatch[1];
    const results = this.loader.getResultsForFile(document.fileName);
    const result = results.find(r => r.testName === funcName);
    if (!result) return undefined;

    const md = new vscode.MarkdownString();
    md.isTrusted = true;
    md.supportThemeIcons = true;

    const passed = result.checks.filter(c => c.passed).length;
    const total = result.checks.length;
    const icon = result.allPassed ? '$(check)' : '$(error)';

    md.appendMarkdown(`### ${icon} checkllm: ${passed}/${total} checks passed\n\n`);
    md.appendMarkdown(`| Check | Score | Status |\n`);
    md.appendMarkdown(`|-------|------:|--------|\n`);

    for (const check of result.checks) {
      const status = check.passed ? '$(check) Pass' : '$(error) Fail';
      md.appendMarkdown(
        `| ${check.metricName} | ${check.score.toFixed(2)} | ${status} |\n`
      );
    }

    if (result.totalCost > 0) {
      md.appendMarkdown(`\n**Cost:** $${result.totalCost.toFixed(4)}\n`);
    }

    return new vscode.Hover(md);
  }
}
