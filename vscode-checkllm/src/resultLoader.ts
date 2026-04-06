import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { CheckResult, SnapshotData, TestResult } from './types';

/**
 * Loads checkllm results from .checkllm/ directory.
 * Reads the latest snapshot JSON file.
 */
export class ResultLoader {
  private results: Map<string, TestResult> = new Map();
  private workspaceRoot: string;

  constructor(workspaceRoot: string) {
    this.workspaceRoot = workspaceRoot;
  }

  /** Load results from the latest snapshot file. */
  loadResults(): Map<string, TestResult> {
    this.results.clear();

    const snapshotDir = path.join(this.workspaceRoot, '.checkllm', 'snapshots');
    if (!fs.existsSync(snapshotDir)) {
      return this.results;
    }

    // Find the latest snapshot
    const files = fs.readdirSync(snapshotDir)
      .filter(f => f.endsWith('.json'))
      .sort()
      .reverse();

    if (files.length === 0) {
      return this.results;
    }

    const latestFile = path.join(snapshotDir, files[0]);
    try {
      const raw = fs.readFileSync(latestFile, 'utf-8');
      const snapshot: SnapshotData = JSON.parse(raw);
      this.parseSnapshot(snapshot);
    } catch (err) {
      console.error('checkllm: failed to load snapshot', err);
    }

    return this.results;
  }

  private parseSnapshot(snapshot: SnapshotData): void {
    for (const [testId, runs] of Object.entries(snapshot.tests)) {
      if (runs.length === 0) continue;

      // Use the latest run
      const latestRun = runs[runs.length - 1];
      const checks: CheckResult[] = [];
      let totalCost = 0;
      let allPassed = true;

      for (const [metricName, metric] of Object.entries(latestRun.metrics)) {
        const check: CheckResult = {
          metricName,
          score: metric.score,
          passed: metric.passed,
          reasoning: '',
          cost: 0,
          latencyMs: 0,
        };
        checks.push(check);
        if (!metric.passed) allPassed = false;
      }

      // Parse test ID: "tests/test_example.py::test_function_name"
      const parts = testId.split('::');
      const filePath = parts[0] || testId;
      const testName = parts[parts.length - 1] || testId;

      this.results.set(testId, {
        testId,
        testName,
        filePath,
        line: 0, // Will be resolved by CodeLens provider
        checks,
        totalCost,
        allPassed,
      });
    }
  }

  /** Get result for a specific test ID. */
  getResult(testId: string): TestResult | undefined {
    return this.results.get(testId);
  }

  /** Get all results. */
  getAllResults(): TestResult[] {
    return Array.from(this.results.values());
  }

  /** Find results matching a file path. */
  getResultsForFile(filePath: string): TestResult[] {
    const relPath = path.relative(this.workspaceRoot, filePath).replace(/\\/g, '/');
    return this.getAllResults().filter(r =>
      r.filePath.replace(/\\/g, '/') === relPath
    );
  }
}
