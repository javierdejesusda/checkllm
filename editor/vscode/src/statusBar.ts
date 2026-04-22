/**
 * Status bar integration for the checkllm extension.
 *
 * Shows the last run's pass/fail count; clicking the item refreshes
 * the snapshot. The snapshot format is the JSON produced by
 * ``checkllm ci --snapshot`` at ``.checkllm/ci_snapshot.json``.
 */

import * as vscode from 'vscode';

interface Metric {
  passed: boolean;
  score: number;
}

interface TestRun {
  metrics: Record<string, Metric>;
}

interface Snapshot {
  version?: number;
  tests: Record<string, TestRun[]>;
}

export class StatusBar implements vscode.Disposable {
  private readonly item: vscode.StatusBarItem;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left, 100);
    this.item.command = 'checkllm.runSuite';
    this.item.text = '$(beaker) checkllm';
    this.item.tooltip = 'Run checkllm suite';
    this.item.show();
  }

  markRunning(label: string): void {
    this.item.text = `$(sync~spin) checkllm: ${label}`;
    this.item.tooltip = `Running ${label}…`;
  }

  async loadSnapshotFromWorkspace(workspaceRoot: vscode.Uri): Promise<void> {
    const snapshotUri = vscode.Uri.joinPath(
        workspaceRoot, '.checkllm', 'ci_snapshot.json');
    await this.updateFromSnapshot(snapshotUri);
  }

  async updateFromSnapshot(uri: vscode.Uri): Promise<void> {
    try {
      const buf = await vscode.workspace.fs.readFile(uri);
      const snap = JSON.parse(Buffer.from(buf).toString('utf8')) as Snapshot;
      const counts = this.countChecks(snap);
      this.item.text =
          `$(beaker) checkllm: ${counts.passed}/${counts.total}`;
      this.item.tooltip =
          `checkllm: ${counts.passed} passed, ${counts.failed} failed`;
    } catch {
      // Snapshot missing or malformed — leave the idle label in place.
    }
  }

  private countChecks(snap: Snapshot):
      {passed: number; failed: number; total: number} {
    let passed = 0;
    let total = 0;
    for (const runs of Object.values(snap.tests ?? {})) {
      for (const run of runs) {
        for (const metric of Object.values(run.metrics ?? {})) {
          total += 1;
          if (metric.passed) {
            passed += 1;
          }
        }
      }
    }
    return {passed, failed: total - passed, total};
  }

  dispose(): void {
    this.item.dispose();
  }
}
