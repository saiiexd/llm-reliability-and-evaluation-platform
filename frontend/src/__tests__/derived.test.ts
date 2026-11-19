import { describe, expect, it } from "vitest";
import {
  formatPercent,
  formatScore,
  formatTimestamp,
  getRagMetadata,
  isRagExecution,
  runCounts,
  shortFingerprint,
  testCasePassed,
  testCaseSucceeded,
} from "../lib/derived";
import type { ModelResponse, RunResult, TestCaseResult } from "../types/domain";

const dummyResponse = (output_text = "ans", rag?: any): ModelResponse => ({
  test_case_id: "t1",
  output_text,
  model_id: "m",
  latency_ms: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  total_tokens: 10,
  provider_metadata: rag ? { rag } : {},
});

describe("Derived Helpers", () => {
  it("determines testCaseSucceeded accurately", () => {
    const successCase: TestCaseResult = {
      test_case_id: "t1",
      input_text: "q",
      reference_answer: "a",
      model_response: dummyResponse("ans"),
      evaluation_results: [],
      execution_error: null,
    };
    expect(testCaseSucceeded(successCase)).toBe(true);

    const errorCase: TestCaseResult = {
      ...successCase,
      execution_error: "Model timed out",
    };
    expect(testCaseSucceeded(errorCase)).toBe(false);
  });

  it("strictly distinguishes execution success from evaluation pass in testCasePassed", () => {
    // Succeeded execution but failed evaluation (e.g. extra text mismatch)
    const execSuccessEvalFail: TestCaseResult = {
      test_case_id: "tc-1",
      input_text: "What is the capital of France?",
      reference_answer: "Paris",
      model_response: dummyResponse("Question: What is the capital of France?\nAnswer: Paris"),
      evaluation_results: [
        {
          evaluator_name: "exact_match",
          test_case_id: "tc-1",
          status: "success",
          criterion: "correctness",
          score: 0.0,
          passed: false,
          label: "mismatch",
          explanation: "Generated output did not exactly match the reference answer.",
          details: {},
          evaluator_config: {},
          error: null,
        },
      ],
      execution_error: null,
    };
    expect(testCaseSucceeded(execSuccessEvalFail)).toBe(true);
    expect(testCasePassed(execSuccessEvalFail)).toBe(false);

    // Succeeded execution and passed evaluation
    const execSuccessEvalPass: TestCaseResult = {
      ...execSuccessEvalFail,
      model_response: dummyResponse("Paris"),
      evaluation_results: [
        {
          ...execSuccessEvalFail.evaluation_results[0],
          score: 1.0,
          passed: true,
          label: "match",
        },
      ],
    };
    expect(testCaseSucceeded(execSuccessEvalPass)).toBe(true);
    expect(testCasePassed(execSuccessEvalPass)).toBe(true);

    // Failed execution with no evaluation
    const execFail: TestCaseResult = {
      test_case_id: "tc-fail",
      input_text: "q",
      reference_answer: "a",
      model_response: null,
      evaluation_results: [],
      execution_error: "Connection timeout",
    };
    expect(testCaseSucceeded(execFail)).toBe(false);
    expect(testCasePassed(execFail)).toBe(false);

    // Multi-evaluator any-fail rule
    const multiEvalSplit: TestCaseResult = {
      test_case_id: "tc-multi",
      input_text: "q",
      reference_answer: "a",
      model_response: dummyResponse("a"),
      evaluation_results: [
        {
          evaluator_name: "semantic_similarity",
          test_case_id: "tc-multi",
          status: "success",
          criterion: "correctness",
          score: 0.95,
          passed: true,
          label: "similar",
          explanation: null,
          details: {},
          evaluator_config: {},
          error: null,
        },
        {
          evaluator_name: "exact_match",
          test_case_id: "tc-multi",
          status: "success",
          criterion: "correctness",
          score: 0.0,
          passed: false,
          label: "mismatch",
          explanation: null,
          details: {},
          evaluator_config: {},
          error: null,
        },
      ],
      execution_error: null,
    };
    expect(testCaseSucceeded(multiEvalSplit)).toBe(true);
    expect(testCasePassed(multiEvalSplit)).toBe(false);
  });

  it("calculates runCounts correctly for both execution and evaluation metrics", () => {
    const mockRunResult: RunResult = {
      dataset_name: "ds",
      model_id: "m",
      evaluator_names: ["exact_match"],
      test_case_results: [
        {
          test_case_id: "t1",
          input_text: "q1",
          reference_answer: "a1",
          model_response: dummyResponse("a1"),
          evaluation_results: [
            {
              evaluator_name: "exact_match",
              test_case_id: "t1",
              status: "success",
              criterion: "correctness",
              score: 1.0,
              passed: true,
              label: "match",
              explanation: null,
              details: {},
              evaluator_config: {},
              error: null,
            },
          ],
          execution_error: null,
        },
        {
          test_case_id: "t2",
          input_text: "q2",
          reference_answer: "a2",
          model_response: dummyResponse("wrong"),
          evaluation_results: [
            {
              evaluator_name: "exact_match",
              test_case_id: "t2",
              status: "success",
              criterion: "correctness",
              score: 0.0,
              passed: false,
              label: "mismatch",
              explanation: null,
              details: {},
              evaluator_config: {},
              error: null,
            },
          ],
          execution_error: null,
        },
        {
          test_case_id: "t3",
          input_text: "q3",
          reference_answer: "a3",
          model_response: null,
          evaluation_results: [],
          execution_error: "Timeout",
        },
      ],
    };
    const counts = runCounts(mockRunResult);
    expect(counts.numTestCases).toBe(3);
    expect(counts.numSucceeded).toBe(2);
    expect(counts.numFailed).toBe(1);
    expect(counts.numPassed).toBe(1);
    expect(counts.numEvalFailed).toBe(2);
    expect(counts.passRate).toBeCloseTo(33.33, 1);
    expect(counts.execSuccessRate).toBeCloseTo(66.67, 1);
  });

  it("detects RAG execution metadata", () => {
    const nonRagResp = dummyResponse("a");
    expect(getRagMetadata(nonRagResp)).toBeNull();

    const ragMeta = {
      query: "q",
      retrieved_chunks: [],
      retrieval_config: {},
      pipeline_config: {},
      prompt: "p",
    };
    const ragResp = dummyResponse("a", ragMeta);
    expect(getRagMetadata(ragResp)).toBeDefined();

    const tc: TestCaseResult = {
      test_case_id: "t1",
      input_text: "q",
      reference_answer: "a",
      model_response: ragResp,
      evaluation_results: [],
      execution_error: null,
    };
    expect(isRagExecution(tc)).toBe(true);
  });

  it("formats score, percent, and fingerprints", () => {
    expect(formatScore(0.8523)).toBe("0.85");
    expect(formatScore(null)).toBe("n/a");
    expect(formatPercent(0.75)).toBe("75%");
    expect(formatPercent(null)).toBe("n/a");
    expect(shortFingerprint("abcdef1234567890", 6)).toBe("abcdef");
    expect(formatTimestamp("2026-09-16T15:00:00Z")).toContain("2026");
  });
});
