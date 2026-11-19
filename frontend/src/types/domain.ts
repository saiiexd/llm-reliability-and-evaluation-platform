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
  output_text?: string | null;
  passed?: boolean;
  succeeded?: boolean;
}

export interface RunResult {
  dataset_name: string;
  model_id: string;
  evaluator_names: string[];
  test_case_results: TestCaseResult[];
  duration_seconds?: number | null;
  num_succeeded?: number;
  num_failed?: number;
  num_passed?: number;
  num_eval_failed?: number;
  pass_rate?: number;
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
  test_cases?: TestCase[];
  test_case_count?: number;
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
  test_case_count?: number;
  runs_count?: number;
  // The list endpoint returns these flat summary fields instead of the
  // nested dataset_version / model_config / evaluator_configs objects.
  dataset_name?: string;
  model_id?: string;
  evaluator_count?: number;
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
  duration_seconds?: number | null;
  num_test_cases?: number;
  num_succeeded?: number;
  num_failed?: number;
  num_passed?: number;
  pass_rate?: number;
  benchmark_id?: string | null;
  repetition_index?: number;
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
  k?: number | null;
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
  assessment?: EvidenceAssessment;
  status?: ClaimSupportStatus;
  method?: string;
  confidence?: number | null;
  evidence_snippets?: string[];
  reasoning?: string | null;
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
  secondary_findings: string[];
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

export type ComparisonCompatibility =
  | "identical_configuration"
  | "partially_different_configuration"
  | "materially_incompatible";

export interface EvaluationAggregate {
  evaluator_name: string;
  criterion: string | null;
  total_cases: number;
  num_evaluated: number;
  num_success: number;
  num_skipped: number;
  num_invalid_config: number;
  num_execution_error: number;
  num_output_validation_error: number;
  num_passed: number;
  num_failed: number;
  pass_rate: number | null;
  pass_rate_numerator: number;
  pass_rate_denominator: number;
  score_count: number;
  score_mean: number | null;
  score_min: number | null;
  score_max: number | null;
  criterion_outcomes: Record<string, number>;
}

export interface FailureCategoryCount {
  category: FailureCategory;
  count: number;
  proportion: number | null;
  denominator: number;
}

export interface SecondaryFindingCount {
  finding_name: string;
  count: number;
  frequency: number | null;
  denominator: number;
}

export interface FailureDistribution {
  total_cases: number;
  num_determined: number;
  num_undetermined: number;
  primary_categories: Record<string, FailureCategoryCount>;
  secondary_findings: Record<string, SecondaryFindingCount>;
  source_run_ids: string[];
}

export interface RetrievalSummary {
  metric_name: string;
  k: number | null;
  total_cases: number;
  evaluated_cases: number;
  skipped_cases: number;
  mean_score: number | null;
  min_score: number | null;
  max_score: number | null;
}

export interface EvaluatorAgreementSummary {
  evaluator_a: string;
  evaluator_b: string;
  criterion: string;
  cases_compared: number;
  agreements: number;
  disagreements: number;
  agreement_rate: number | null;
  excluded_cases: number;
  disagreement_test_case_ids: string[];
}

export interface RobustnessSummary {
  total_pairs: number;
  num_unchanged: number;
  num_changed: number;
  consistency_rate: number | null;
  by_perturbation: Record<
    string,
    { total: number; unchanged: number; changed: number; rate: number | null }
  >;
  transition_counts: Record<string, number>;
}

export interface ExecutionPerformanceSummary {
  run_id: string;
  wall_clock_duration_seconds: number | null;
  per_case_duration_seconds_mean: number | null;
  retrieval_duration_seconds_mean: number | null;
  generation_duration_seconds_mean: number | null;
  evaluator_duration_seconds_mean: number | null;
  tokens_available: boolean;
  cost_available: boolean;
  notes: string[];
  prompt_tokens_total?: number | null;
  completion_tokens_total?: number | null;
  total_tokens?: number | null;
}

export interface EvaluationComparisonMetric {
  pass_rates: Record<string, number | null>;
  numerators: Record<string, number>;
  denominators: Record<string, number>;
  score_means: Record<string, number | null>;
}

export interface RunComparison {
  source_run_ids: string[];
  compatibility: ComparisonCompatibility;
  compatibility_reasons: string[];
  configuration_diffs: Record<string, unknown>;
  evaluation_comparisons: Record<string, EvaluationComparisonMetric>;
  failure_distribution_comparisons: Record<string, Record<string, number>>;
  retrieval_comparisons: Record<string, Record<string, number | null>>;
  robustness_comparisons: Record<string, number | null>;
  performance_comparisons: Record<string, Record<string, number | null>>;
}

export interface TrendAnalysis {
  run_ids: string[];
  experiment_id: string | null;
  is_repeated_run_trend: boolean;
  has_trend: boolean;
  configuration_transitions: number[];
  metric_series: Record<string, (number | null)[]>;
  sample_sizes: number[];
  timestamps: string[];
}

export interface RecentRunSummary {
  run_id: string;
  experiment_id: string;
  started_at: string;
  status: string;
  case_count: number;
  config_fingerprint: string;
}

export interface AnalyticsOverview {
  total_experiments: number;
  total_runs: number;
  total_test_cases: number;
  total_evaluations: number;
  evaluator_coverage: Record<string, number>;
  disagreement_count: number;
  primary_failure_distribution: Record<string, number>;
  retrieval_runs_count: number;
  mean_retrieval_mrr: number | null;
  recent_runs: RecentRunSummary[];
}

export interface ProviderModelInfo {
  model_id: string;
  display_name: string;
  context_window: number;
  default_temperature: number;
}

export interface ProviderStatus {
  provider_id: string;
  name: string;
  is_configured: boolean;
  base_url: string | null;
  requires_api_key: boolean;
  supported_models: ProviderModelInfo[];
  default_model: string;
}

export interface ProvidersListResponse {
  providers: ProviderStatus[];
}

export interface BenchmarkExecutionRequest {
  experiment_id: string;
  adapter_ids: string[];
  repetitions?: number;
  allow_external_inference?: boolean;
  description?: string;
}

export interface BenchmarkExecutionResponse {
  benchmark_id: string;
  experiment_id: string;
  total_runs: number;
  runs: {
    run_id: string;
    experiment_id: string;
    status: string;
    started_at?: string;
    finished_at?: string;
    duration_seconds?: number | null;
    num_succeeded: number;
    num_failed: number;
    error?: string | null;
  }[];
}

