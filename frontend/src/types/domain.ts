/**
 * Domain types mirroring the Python backend's persisted JSON shapes exactly
 * (see `to_dict()` on each corresponding dataclass in `src/llm_reliability`).
 * These are hand-written to match the backend's serialization format, not
 * generated, and must be kept in sync manually when the backend's `to_dict`
 * shapes change.
 *
 * Note: Python `@property` values (for example `TestCaseResult.succeeded`,
 * `RunResult.num_succeeded`) are NOT part of `to_dict()` output, since
 * `dataclasses.asdict` only serializes actual fields. Equivalent derived
 * values are computed on the frontend in `lib/derived.ts` instead of being
 * duplicated here as if they were persisted fields.
 */

export type EvaluationStatus =
  | "success"
  | "skipped"
  | "invalid_configuration"
  | "execution_error"
  | "output_validation_error";

export interface EvaluationResult {
  evaluator_name: string;
  test_case_id: string;
  status: EvaluationStatus;
  criterion: string | null;
  score: number | null;
  passed: boolean | null;
  label: string | null;
  explanation: string | null;
  details: Record<string, unknown>;
  evaluator_config: Record<string, unknown>;
  error: string | null;
}

export interface RagRetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  rank: number;
}

export interface RagMetadata {
  query: string;
  retrieved_chunks: RagRetrievedChunk[];
  retrieval_config: Record<string, unknown> | null;
  pipeline_config: Record<string, unknown>;
  prompt: string;
}

export interface ModelResponse {
  test_case_id: string;
  output_text: string;
  model_id: string;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  provider_metadata: Record<string, unknown> & { rag?: RagMetadata };
}

export interface TestCaseResult {
  test_case_id: string;
  input_text: string;
  reference_answer: string | null;
  model_response: ModelResponse | null;
  evaluation_results: EvaluationResult[];
  execution_error: string | null;
}

export interface RunResult {
  dataset_name: string;
  model_id: string;
  evaluator_names: string[];
  test_case_results: TestCaseResult[];
}

export interface TestCase {
  id: string;
  input: string;
  reference_answer: string | null;
  metadata: Record<string, unknown>;
}

export interface DatasetVersionData {
  dataset_name: string;
  version_id: string;
  test_cases: TestCase[];
}

export interface ModelConfigData {
  model_id: string;
  temperature: number | null;
  max_tokens: number | null;
  extra_params: Record<string, unknown>;
}

export interface EvaluatorConfigData {
  evaluator_name: string;
  parameters: Record<string, unknown>;
}

export interface ExperimentDefinition {
  schema_version: number;
  experiment_id: string;
  name: string;
  description: string | null;
  dataset_version: DatasetVersionData;
  model_config: ModelConfigData;
  evaluator_configs: EvaluatorConfigData[];
  metadata: Record<string, unknown>;
  created_at: string;
  config_fingerprint: string;
}

export type RunStatus = "succeeded" | "partially_succeeded" | "failed";

export interface ExperimentRun {
  schema_version: number;
  run_id: string;
  experiment_id: string;
  status: RunStatus;
  started_at: string;
  finished_at: string;
  dataset_version: DatasetVersionData;
  model_config: ModelConfigData;
  evaluator_configs: EvaluatorConfigData[];
  config_fingerprint: string;
  result: RunResult | null;
  error: string | null;
}

export type RetrievalEvaluationStatus = "success" | "skipped" | "execution_error";

export interface RetrievalEvaluationResult {
  evaluator_name: string;
  test_case_id: string;
  metric_name: string;
  status: RetrievalEvaluationStatus;
  score: number | null;
  relevant_targets: string[] | null;
  retrieved_ids: string[];
  explanation: string | null;
  error: string | null;
  evaluator_config: Record<string, unknown>;
}

export type DiagnosticStatus = "determined" | "insufficient_evidence" | "undetermined" | "error";

export type FailureCategory =
  | "correct"
  | "incorrect"
  | "unsupported"
  | "contradicted"
  | "retrieval_failure"
  | "generation_failure"
  | "evaluator_disagreement"
  | "insufficient_evidence"
  | "undetermined";

export type ClaimSupportStatus = "supported" | "unsupported" | "contradicted" | "undetermined";

export interface EvidenceAssessment {
  status: ClaimSupportStatus;
  method: string;
  supporting_evidence: string[];
  confidence: number | null;
  explanation: string | null;
}

export interface ClaimAssessment {
  statement: string;
  assessment: EvidenceAssessment;
}

export interface ReliabilityDiagnostic {
  test_case_id: string;
  question: string;
  generated_answer: string | null;
  reference_answer: string | null;
  retrieved_context: string[] | null;
  status: DiagnosticStatus;
  failure_category: FailureCategory | null;
  contributing_evaluators: string[];
  supporting_evidence: string[];
  confidence: number | null;
  explanation: string;
  claim_assessments: ClaimAssessment[];
  error: string | null;
}

export type PerturbationType =
  | "question_rewording"
  | "context_reordering"
  | "top_k_change"
  | "irrelevant_context_injection"
  | "context_modification"
  | "repeated_input";

export interface RobustnessResult {
  baseline_test_case_id: string;
  perturbed_test_case_id: string;
  perturbation_type: PerturbationType;
  baseline_category: string | null;
  perturbed_category: string | null;
  outcome_changed: boolean;
  failure_category_transition: string | null;
}

export interface ConsistencyResult {
  num_comparisons: number;
  num_unchanged: number;
  outcome_consistency_rate: number | null;
  transition_counts: Record<string, number>;
}

export interface EvaluatorSummary {
  evaluator_name: string;
  criterion: string | null;
  num_evaluated: number;
  num_success: number;
  num_skipped: number;
  num_error: number;
  num_passed: number;
  num_failed: number;
  score_count: number;
  score_mean: number | null;
  score_min: number | null;
  score_max: number | null;
}

export interface EvaluationSummary {
  dataset_name: string;
  model_id: string;
  num_test_cases: number;
  num_execution_succeeded: number;
  num_execution_failed: number;
  per_evaluator: Record<string, EvaluatorSummary>;
  disagreement_count: number;
}

export type ReliabilityFlagType =
  | "evaluator_disagreement"
  | "missing_evaluation_evidence"
  | "evaluator_failure"
  | "incorrect_per_evaluator";

export interface ReliabilityFinding {
  test_case_id: string;
  flag_type: ReliabilityFlagType;
  detail: string;
  evaluator_names: string[];
}

export interface RetrievalEvaluatorSummary {
  evaluator_name: string;
  metric_name: string;
  num_evaluated: number;
  num_success: number;
  num_skipped: number;
  num_error: number;
  score_mean: number | null;
  score_min: number | null;
  score_max: number | null;
}

export interface RunSummaryBundle {
  evaluation_summary: EvaluationSummary;
  reliability_findings: ReliabilityFinding[];
  retrieval_summary: Record<string, RetrievalEvaluatorSummary>;
}

export interface RunRobustnessBundle {
  comparisons: RobustnessResult[];
  consistency: ConsistencyResult;
}

export interface FixtureMeta {
  generated_by: string;
  is_fixture_data: true;
  warning: string;
  experiment_ids: string[];
  run_ids: string[];
}
