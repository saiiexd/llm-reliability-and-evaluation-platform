/**
 * Small derived-value helpers.
 *
 * The Python backend's `@property` values (`TestCaseResult.succeeded`,
 * `RunResult.num_succeeded`, etc.) are not part of persisted JSON --
 * `dataclasses.asdict` only serializes actual fields. These functions
 * recompute the same logic on the frontend from the raw fields, rather
 * than inventing new derived semantics that might drift from the backend.
 */
import type { ExperimentDefinition, ModelResponse, RunResult, TestCaseResult } from "../types/domain";

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

/**
 * One plain-language outcome per test case.
 *
 * The backend's `passed` flag is false both when an evaluator judged the
 * answer wrong AND when an evaluator could not produce a verdict (e.g. the
 * LLM judge returned invalid output). Showing both as "Failed" made correct
 * answers look wrong. This splits "not passed" into two honest states
 * without changing which cases count as passed:
 *
 *   passed   - every check ran and passed (same set as testCasePassed)
 *   failed   - at least one check ran and judged the answer wrong
 *   unscored - no check said wrong, but at least one could not give a verdict
 *   error    - the model call itself crashed, so there is no answer to check
 */
export type Verdict = "passed" | "failed" | "unscored" | "error";

export function testCaseVerdict(result: TestCaseResult): Verdict {
  if (!testCaseSucceeded(result)) return "error";
  const evals = result.evaluation_results ?? [];
  if (evals.some((e) => e.passed === false)) return "failed";
  if (testCasePassed(result)) return "passed";
  return "unscored";
}

/** Outcome of a single evaluator check on one test case. */
export function evaluatorVerdict(e: { passed: boolean | null }): "passed" | "failed" | "unscored" {
  if (e.passed === true) return "passed";
  if (e.passed === false) return "failed";
  return "unscored";
}

export function verdictCounts(results: TestCaseResult[]): Record<Verdict, number> & { total: number } {
  const counts = { passed: 0, failed: 0, unscored: 0, error: 0, total: results.length };
  for (const r of results) counts[testCaseVerdict(r)] += 1;
  return counts;
}

/**
 * Passed / total for a run, from whichever shape we have: list endpoints
 * return summaries (num_passed, num_test_cases) without per-check detail,
 * full runs carry test_case_results.
 */
export function runPassCounts(run: {
  result?: RunResult | null;
  num_passed?: number;
  num_test_cases?: number;
}): { passed: number; total: number } {
  if (run.result) {
    return {
      passed: run.result.test_case_results.filter(testCasePassed).length,
      total: run.result.test_case_results.length,
    };
  }
  return { passed: run.num_passed ?? 0, total: run.num_test_cases ?? 0 };
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  passed: "Passed",
  failed: "Wrong",
  unscored: "Not scored",
  error: "Crashed",
};

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

/** Plain-language meaning of each diagnosed failure reason. */
export const CATEGORY_HELP: Record<string, string> = {
  correct: "The answer was judged correct.",
  incorrect: "The answer didn't match the expected answer.",
  retrieval_failure: "The search step didn't find the passage that contains the answer.",
  generation_failure: "The right passage was found, but the model still answered wrongly.",
  unsupported: "The answer says things the retrieved passages don't support.",
  contradicted: "The answer contradicts the retrieved passages.",
  evaluator_disagreement: "The checks disagreed about whether the answer is right.",
  insufficient_evidence: "There wasn't enough evidence to decide.",
  undetermined: "The evidence doesn't point to a single cause.",
};

const EVALUATOR_INFO: Record<string, { label: string; help: string }> = {
  exact_match: {
    label: "Exact match",
    help: "Passes if the answer text matches the expected answer exactly.",
  },
  llm_faithfulness_judge: {
    label: "Faithfulness (LLM judge)",
    help: "A second model checks that the answer is backed by the retrieved passages.",
  },
};

export function evaluatorLabel(name: string): string {
  return EVALUATOR_INFO[name]?.label ?? name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function evaluatorHelp(name: string): string | null {
  return EVALUATOR_INFO[name]?.help ?? null;
}

/** Fields that differ between the experiment list (flat) and detail (nested) shapes. */
export function experimentModel(e: ExperimentDefinition): string | null {
  return e.model_config?.model_id ?? e.model_id ?? null;
}

export function experimentCaseCount(e: ExperimentDefinition): number {
  return e.dataset_version?.test_case_count ?? e.dataset_version?.test_cases?.length ?? e.test_case_count ?? 0;
}

export function experimentDataset(e: ExperimentDefinition): string | null {
  return e.dataset_version?.dataset_name ?? e.dataset_name ?? null;
}
