import { describe, expect, it } from "vitest";
import {
  toneForClaimSupport,
  toneForDiagnosticStatus,
  toneForEvaluationStatus,
  toneForFailureCategory,
  toneForRunStatus,
} from "../lib/tone";

describe("Tone Mappings", () => {
  it("maps failure categories to intended tones", () => {
    expect(toneForFailureCategory("correct")).toBe("neutral");
    expect(toneForFailureCategory("incorrect")).toBe("negative");
    expect(toneForFailureCategory("contradicted")).toBe("negative");
    expect(toneForFailureCategory("retrieval_failure")).toBe("negative");
    expect(toneForFailureCategory("generation_failure")).toBe("negative");
    expect(toneForFailureCategory("unsupported")).toBe("caution");
    expect(toneForFailureCategory("evaluator_disagreement")).toBe("caution");
    expect(toneForFailureCategory("insufficient_evidence")).toBe("muted");
    expect(toneForFailureCategory("undetermined")).toBe("muted");
    expect(toneForFailureCategory(null)).toBe("muted");
  });

  it("maps diagnostic and run statuses", () => {
    expect(toneForDiagnosticStatus("determined")).toBe("neutral");
    expect(toneForDiagnosticStatus("error")).toBe("negative");
    expect(toneForDiagnosticStatus("undetermined")).toBe("muted");

    expect(toneForRunStatus("succeeded")).toBe("neutral");
    expect(toneForRunStatus("partially_succeeded")).toBe("caution");
    expect(toneForRunStatus("failed")).toBe("negative");
  });

  it("maps evaluation and claim statuses", () => {
    expect(toneForEvaluationStatus("success")).toBe("neutral");
    expect(toneForEvaluationStatus("execution_error")).toBe("negative");
    expect(toneForEvaluationStatus("skipped")).toBe("muted");

    expect(toneForClaimSupport("supported")).toBe("neutral");
    expect(toneForClaimSupport("contradicted")).toBe("negative");
    expect(toneForClaimSupport("unsupported")).toBe("caution");
  });
});

