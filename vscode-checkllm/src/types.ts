/** Result for a single check within a test. */
export interface CheckResult {
  metricName: string;
  score: number;
  passed: boolean;
  reasoning: string;
  cost: number;
  latencyMs: number;
}

/** Aggregated results for a single test function. */
export interface TestResult {
  testId: string;
  testName: string;
  filePath: string;
  line: number;
  checks: CheckResult[];
  totalCost: number;
  allPassed: boolean;
}

/** Snapshot data from .checkllm/ JSON files. */
export interface SnapshotData {
  version: number;
  timestamp: string;
  tests: Record<string, SnapshotTestRun[]>;
}

export interface SnapshotTestRun {
  metrics: Record<string, { score: number; passed: boolean }>;
}
