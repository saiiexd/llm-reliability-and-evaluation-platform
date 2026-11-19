import { describe, expect, it } from "vitest";
import { testCaseVerdict, verdictCounts } from "../lib/derived";
import type { EvaluationResult, TestCaseResult } from "../types/domain";

const evalResult = (passed: boolean | null, status: EvaluationResult["status"] = "success"): EvaluationResult => ({
  evaluator_name: passed === null ? "llm_faithfulness_judge" : "exact_match",
  test_case_id: "tc",
  status,
  criterion: "correctness",
  score: passed === null ? null : passed ? 1 : 0,
  passed,
  label: null,
  explanation: null,
  details: {},
  evaluator_config: {},
  error: passed === null ? "judge returned invalid JSON" : null,
});

const testCase = (evals: EvaluationResult[], extra: Partial<TestCaseResult> = {}): TestCaseResult => ({
  test_case_id: "tc",
  input_text: "What is the capital of France?",
  reference_answer: "Paris",
  output_text: "Paris",
  model_response: null,
  evaluation_results: evals,
  execution_error: null,
  ...extra,
});

describe("testCaseVerdict", () => {
  it("is passed when every check ran and passed", () => {
    expect(testCaseVerdict(testCase([evalResult(true)]))).toBe("passed");
  });

  it("is failed when any check judged the answer wrong", () => {
    expect(testCaseVerdict(testCase([evalResult(false), evalResult(null, "output_validation_error")]))).toBe("failed");
  });

  it("is unscored, not failed, when a correct answer could not be fully checked", () => {
    // Real case from the demo data: exact match passed, the LLM judge errored,
    // and the backend reports passed=false.
    const tc = testCase([evalResult(true), evalResult(null, "output_validation_error")], { passed: false });
    expect(testCaseVerdict(tc)).toBe("unscored");
  });

  it("is unscored when no evaluators ran", () => {
    expect(testCaseVerdict(testCase([]))).toBe("unscored");
  });

  it("is error when the model call crashed", () => {
    expect(testCaseVerdict(testCase([evalResult(true)], { execution_error: "timeout" }))).toBe("error");
  });

  it("counts each verdict", () => {
    const counts = verdictCounts([
      testCase([evalResult(true)]),
      testCase([evalResult(false)]),
      testCase([evalResult(true), evalResult(null, "output_validation_error")]),
    ]);
    expect(counts).toEqual({ passed: 1, disputed: 0, failed: 1, unscored: 1, error: 0, total: 3 });
  });
});

describe("disputed verdict", () => {
  const ev = (name: string, criterion: string, passed: boolean): EvaluationResult => ({
    ...evalResult(passed),
    evaluator_name: name,
    criterion,
    error: null,
  });

  it("is disputed when two checks of the same criterion disagree", () => {
    const tc = testCase([ev("exact_match", "correctness", false), ev("llm_judge", "correctness", true)]);
    expect(testCaseVerdict(tc)).toBe("disputed");
  });

  it("stays failed when checks of different criteria differ", () => {
    const tc = testCase([ev("exact_match", "correctness", true), ev("llm_faithfulness_judge", "faithfulness", false)]);
    expect(testCaseVerdict(tc)).toBe("failed");
  });
});
