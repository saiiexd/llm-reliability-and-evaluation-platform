/**
 * Maps backend status/category values to a small, restrained set of visual
 * "tones," used by <Tag> everywhere a category appears. Deliberately not
 * one distinct bright color per category: the accent color is reserved for
 * genuine failures/contradictions, and uncertainty is communicated by
 * style (dashed border, muted text), not by yet another hue. A tag is
 * always paired with its label text -- color is never the only signal.
 */
import type {
  ClaimSupportStatus,
  DiagnosticStatus,
  EvaluationStatus,
  FailureCategory,
  RetrievalEvaluationStatus,
  RunStatus,
} from "../types/domain";

export type Tone = "neutral" | "negative" | "caution" | "muted";

export function toneForFailureCategory(category: FailureCategory | null): Tone {
  switch (category) {
    case "correct":
      return "neutral";
    case "incorrect":
    case "contradicted":
    case "retrieval_failure":
    case "generation_failure":
      return "negative";
    case "unsupported":
    case "evaluator_disagreement":
      return "caution";
    case "insufficient_evidence":
    case "undetermined":
      return "muted";
    default:
      return "muted";
  }
}

export function toneForDiagnosticStatus(status: DiagnosticStatus): Tone {
  switch (status) {
    case "determined":
      return "neutral";
    case "error":
      return "negative";
    case "undetermined":
    case "insufficient_evidence":
      return "muted";
  }
}

export function toneForRunStatus(status: RunStatus): Tone {
  switch (status) {
    case "succeeded":
      return "neutral";
    case "partially_succeeded":
      return "caution";
    case "failed":
      return "negative";
  }
}

export function toneForEvaluationStatus(status: EvaluationStatus | RetrievalEvaluationStatus): Tone {
  switch (status) {
    case "success":
      return "neutral";
    case "execution_error":
    case "output_validation_error":
      return "negative";
    case "invalid_configuration":
      return "caution";
    case "skipped":
      return "muted";
    default:
      return "muted";
  }
}

export function toneForClaimSupport(status: ClaimSupportStatus): Tone {
  switch (status) {
    case "supported":
      return "neutral";
    case "contradicted":
      return "negative";
    case "unsupported":
      return "caution";
    case "undetermined":
      return "muted";
  }
}

export const TONE_CLASSES: Record<Tone, string> = {
  neutral: "border-line bg-paper-raised text-ink",
  negative: "border-accent/40 bg-accent-soft text-accent",
  caution: "border-line text-ink",
  muted: "border-dashed border-ink-soft/50 text-ink-soft",
};
