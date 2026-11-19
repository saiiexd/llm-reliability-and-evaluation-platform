/**
 * Small derived-value helpers.
 *
 * The Python backend's `@property` values (`TestCaseResult.succeeded`,
 * `RunResult.num_succeeded`, etc.) are not part of persisted JSON --
 * `dataclasses.asdict` only serializes actual fields. These functions
 * recompute the same logic on the frontend from the raw fields, rather
 * than inventing new derived semantics that might drift from the backend.
 */
import type { ModelResponse, RunResult, TestCaseResult } from "../types/domain";

export function testCaseSucceeded(result: TestCaseResult): boolean {
  if (result.execution_error !== null && result.execution_error !== undefined) {
    return false;
  }
  if (typeof result.succeeded === "boolean") {
    return result.succeeded;
  }
  return result.model_response !== null || (result.output_text !== null && result.output_text !== undefined);
}

export function testCasePassed(result: TestCaseResult): boolean {
  if (!testCaseSucceeded(result)) {
    return false;
  }
  if (typeof result.passed === "boolean") {
    return result.passed;
  }
  const evals = result.evaluation_results ?? [];
  if (evals.length === 0) {
    return false;
  }
  if (evals.some((e) => e.passed === false)) {
    return false;
  }
  const errorStatuses = ["execution_error", "output_validation_error", "invalid_configuration"];
  if (evals.some((e) => e.status && errorStatuses.includes(e.status))) {
    return false;
  }
  return evals.some((e) => e.passed === true);
}

export function runCounts(result: RunResult): {
  numTestCases: number;
  numSucceeded: number;
  numFailed: number;
  numPassed: number;
  numEvalFailed: number;
  passRate: number;
  execSuccessRate: number;
} {
  const numTestCases = result.test_case_results.length;
  const numSucceeded = result.test_case_results.filter(testCaseSucceeded).length;
  const numFailed = numTestCases - numSucceeded;
  const numPassed = result.test_case_results.filter(testCasePassed).length;
  const numEvalFailed = numTestCases - numPassed;
  const passRate = numTestCases > 0 ? (numPassed / numTestCases) * 100 : 0;
  const execSuccessRate = numTestCases > 0 ? (numSucceeded / numTestCases) * 100 : 0;
  return {
    numTestCases,
    numSucceeded,
    numFailed,
    numPassed,
    numEvalFailed,
    passRate,
    execSuccessRate,
  };
}

export function getRagMetadata(response: ModelResponse | null) {
  if (!response) return null;
  return response.provider_metadata.rag ?? null;
}

export function isRagExecution(result: TestCaseResult): boolean {
  return getRagMetadata(result.model_response) !== null;
}

export function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatPercent(value: number | null, digits = 0): string {
  if (value === null || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number | null, digits = 2): string {
  if (value === null || Number.isNaN(value)) return "n/a";
  return value.toFixed(digits);
}

export function shortFingerprint(fingerprint: string, length = 12): string {
  return fingerprint.slice(0, length);
}

export function formatCategoryLabel(category: string): string {
  const map: Record<string, string> = {
    correct: "Correct Answer",
    incorrect: "Incorrect Answer",
    unsupported: "Unsupported Answer",
    contradicted: "Contradicted Answer",
    retrieval_failure: "Retrieval Failure",
    generation_failure: "Generation Failure",
    evaluator_disagreement: "Evaluator Disagreement",
    insufficient_evidence: "Insufficient Evidence",
    undetermined: "Undetermined Failure",
  };
  return (
    map[category.toLowerCase()] ??
    category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}
