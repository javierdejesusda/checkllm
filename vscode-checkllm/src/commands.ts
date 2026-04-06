import * as vscode from 'vscode';
import * as cp from 'child_process';
import { TestResult } from './types';

/** Run checkllm tests in a terminal. */
export function runCheckllm(): void {
  const terminal = vscode.window.createTerminal('checkllm');
  const config = vscode.workspace.getConfiguration('checkllm');
  const budget = config.get<number>('budget', 5.0);

  terminal.show();
  terminal.sendText(`checkllm run tests/ --budget ${budget}`);
}

/** Estimate checkllm costs in a terminal. */
export function estimateCheckllm(): void {
  const terminal = vscode.window.createTerminal('checkllm');
  terminal.show();
  terminal.sendText('checkllm estimate tests/');
}

/** Show check details in an output channel. */
export function showDetails(result: TestResult): void {
  const channel = vscode.window.createOutputChannel('checkllm');
  channel.clear();
  channel.appendLine(`Test: ${result.testName}`);
  channel.appendLine(`File: ${result.filePath}`);
  channel.appendLine(`Status: ${result.allPassed ? 'PASSED' : 'FAILED'}`);
  channel.appendLine(`Cost: $${result.totalCost.toFixed(4)}`);
  channel.appendLine('');
  channel.appendLine('Checks:');
  for (const check of result.checks) {
    const icon = check.passed ? 'PASS' : 'FAIL';
    channel.appendLine(`  [${icon}] ${check.metricName}: ${check.score.toFixed(2)}`);
    if (check.reasoning) {
      channel.appendLine(`         ${check.reasoning}`);
    }
  }
  channel.show();
}
