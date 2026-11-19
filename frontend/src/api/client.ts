/**
 * Typed HTTP Client for the LLM Reliability & Evaluation Platform REST API (/api/v1).
 *
 * Communicates with the FastAPI backend, normalizes error envelopes,
 * injects request timeouts, and preserves response correlation headers.
 */

import type {
  AnalyticsOverview,
  BenchmarkExecutionRequest,
  BenchmarkExecutionResponse,
  EvaluationAggregate,
  EvaluatorAgreementSummary,
  ExecutionPerformanceSummary,
  ExperimentDefinition,
  ExperimentRun,
  FailureDistribution,
  ProvidersListResponse,
  ReliabilityDiagnostic,
  RetrievalEvaluationResult,
  RetrievalSummary,
  RobustnessSummary,
  RunComparison,
  RunRobustnessBundle,
  RunSummaryBundle,
  TestCaseResult,
  TrendAnalysis,
} from "../types/domain";

export interface PaginationParams {
  limit?: number;
  offset?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    has_next: boolean;
    has_previous: boolean;
  };
}

export interface HealthResponse {
  status: "healthy" | "degraded";
  version: string;
  environment: string;
  timestamp: string;
  database: string;
  allowlisted_adapters: string[];
}

export interface RunExecutionRequest {
  adapter_id: string;
  run_id?: string;
  allow_external_inference?: boolean;
  model_config?: Record<string, unknown>;
}

export interface RunExecutionResponse {
  run_id: string;
  experiment_id: string;
  status: string;
  duration_seconds: number;
  test_cases_count: number;
  num_succeeded: number;
  num_failed: number;
  created_at: string;
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: unknown[];
  readonly requestId?: string;

  constructor(
    message: string,
    status: number,
    code: string = "INTERNAL_ERROR",
    details: unknown[] = [],
    requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(
    baseUrl: string = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1",
    timeoutMs: number = 10000,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    const headers: Record<string, string> = {
      Accept: "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (options.body && typeof options.body === "string") {
      headers["Content-Type"] = "application/json";
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      const requestId = response.headers.get("x-request-id") || undefined;

      if (!response.ok) {
        let errBody: { error?: { code?: string; message?: string; details?: unknown[] } } = {};
        try {
          errBody = await response.json();
        } catch {
          // If response body is not JSON
        }

        const code = errBody.error?.code || `HTTP_${response.status}`;
        const message =
          errBody.error?.message ||
          `API request failed with status ${response.status} (${response.statusText})`;
        const details = errBody.error?.details || [];

        throw new ApiError(message, response.status, code, details, requestId);
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        throw err;
      }
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(
          `Request timeout after ${this.timeoutMs}ms while connecting to ${url}`,
          408,
          "TIMEOUT_ERROR",
        );
      }
      const message = err instanceof Error ? err.message : String(err);
      throw new ApiError(
        `Network or connection error connecting to API: ${message}`,
        0,
        "CONNECTION_ERROR",
      );
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // --- Health ---
  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  // --- Experiments ---
  async listExperiments(
    params?: PaginationParams & { search?: string },
  ): Promise<PaginatedResponse<ExperimentDefinition>> {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    if (params?.search) q.set("search", params.search);
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<PaginatedResponse<ExperimentDefinition>>(`/experiments${qs}`);
  }

  async getExperiment(experimentId: string): Promise<ExperimentDefinition> {
    return this.request<ExperimentDefinition>(`/experiments/${encodeURIComponent(experimentId)}`);
  }

  async createExperiment(payload: Record<string, unknown>): Promise<ExperimentDefinition> {
    return this.request<ExperimentDefinition>("/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async listExperimentRuns(
    experimentId: string,
    params?: PaginationParams,
  ): Promise<PaginatedResponse<ExperimentRun>> {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<PaginatedResponse<ExperimentRun>>(
      `/experiments/${encodeURIComponent(experimentId)}/runs${qs}`,
    );
  }

  async executeExperiment(
    experimentId: string,
    payload: RunExecutionRequest,
  ): Promise<RunExecutionResponse> {
    return this.request<RunExecutionResponse>(
      `/experiments/${encodeURIComponent(experimentId)}/execute`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
  }

  // --- Providers & Benchmarks ---
  async listProviders(): Promise<ProvidersListResponse> {
    return this.request<ProvidersListResponse>("/providers");
  }

  async executeBenchmark(
    payload: BenchmarkExecutionRequest,
  ): Promise<BenchmarkExecutionResponse> {
    return this.request<BenchmarkExecutionResponse>("/benchmarks/execute", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async getBenchmarkRuns(benchmarkId: string): Promise<ExperimentRun[]> {
    return this.request<ExperimentRun[]>(
      `/benchmarks/${encodeURIComponent(benchmarkId)}/runs`,
    );
  }

  // --- Runs ---
  async listRuns(
    params?: PaginationParams & { experimentId?: string },
  ): Promise<PaginatedResponse<ExperimentRun>> {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    if (params?.experimentId) q.set("experiment_id", params.experimentId);
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<PaginatedResponse<ExperimentRun>>(`/runs${qs}`);
  }

  async getRun(runId: string, includeTestCases: boolean = true): Promise<ExperimentRun> {
    const qs = includeTestCases ? "?include_test_cases=true" : "?include_test_cases=false";
    const raw = await this.request<any>(`/runs/${encodeURIComponent(runId)}${qs}`);
    if (raw && !raw.result && Array.isArray(raw.test_case_results)) {
      const normalizedCases: TestCaseResult[] = raw.test_case_results.map((tc: any) => {
        if (!tc.model_response && tc.output_text !== undefined) {
          tc.model_response = {
            output_text: tc.output_text,
            model_id: tc.model_id ?? null,
            // Generation latency as recorded by the adapter; null when not
            // reported (mock adapters). Not derived from wall-clock duration,
            // which also includes evaluation time.
            latency_ms: tc.latency_ms ?? null,
            prompt_tokens: tc.prompt_tokens ?? null,
            completion_tokens: tc.completion_tokens ?? null,
            total_tokens: tc.total_tokens ?? null,
            provider_metadata: tc.provider_metadata || {},
          };
        }
        if (!tc.evaluation_results && Array.isArray(tc.evaluations)) {
          tc.evaluation_results = tc.evaluations;
        }
        return tc;
      });
      raw.result = {
        dataset_name: raw.dataset_version?.dataset_name || "",
        model_id: raw.model_config?.model_id || "",
        evaluator_names: raw.evaluator_configs?.map((e: any) => e.evaluator_name) || [],
        duration_seconds: raw.duration_seconds ?? null,
        num_succeeded: raw.num_succeeded,
        num_failed: raw.num_failed,
        num_passed: raw.num_passed,
        pass_rate: raw.pass_rate,
        test_case_results: normalizedCases,
      };
    }
    return raw as ExperimentRun;
  }

  async listTestCases(
    runId: string,
    params?: PaginationParams & { passed?: boolean },
  ): Promise<PaginatedResponse<TestCaseResult>> {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    if (params?.passed !== undefined) q.set("passed", String(params.passed));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<PaginatedResponse<TestCaseResult>>(
      `/runs/${encodeURIComponent(runId)}/test-cases${qs}`,
    );
  }

  async getTestCase(runId: string, testCaseId: string): Promise<TestCaseResult> {
    const tc = await this.request<any>(
      `/runs/${encodeURIComponent(runId)}/test-cases/${encodeURIComponent(testCaseId)}`,
    );
    if (tc && !tc.model_response && tc.output_text !== undefined) {
      tc.model_response = {
        output_text: tc.output_text,
        model_id: tc.model_id ?? null,
        latency_ms: tc.latency_ms ?? null,
        prompt_tokens: tc.prompt_tokens ?? null,
        completion_tokens: tc.completion_tokens ?? null,
        total_tokens: tc.total_tokens ?? null,
        provider_metadata: tc.provider_metadata || {},
      };
    }
    if (tc && !tc.evaluation_results && Array.isArray(tc.evaluations)) {
      tc.evaluation_results = tc.evaluations;
    }
    return tc as TestCaseResult;
  }

  async getRunSummaries(runId: string): Promise<RunSummaryBundle> {
    return this.request<RunSummaryBundle>(`/runs/${encodeURIComponent(runId)}/summaries`);
  }

  // --- Diagnostics ---
  async listDiagnostics(params?: {
    runId?: string;
    testCaseId?: string;
    failureCategory?: string;
    secondaryFinding?: string;
    limit?: number;
    offset?: number;
  }): Promise<PaginatedResponse<ReliabilityDiagnostic>> {
    const q = new URLSearchParams();
    if (params?.runId) q.set("run_id", params.runId);
    if (params?.testCaseId) q.set("test_case_id", params.testCaseId);
    if (params?.failureCategory) q.set("failure_category", params.failureCategory);
    if (params?.secondaryFinding) q.set("secondary_finding", params.secondaryFinding);
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<PaginatedResponse<ReliabilityDiagnostic>>(`/diagnostics${qs}`);
  }

  async getDiagnostic(runId: string, testCaseId: string): Promise<ReliabilityDiagnostic> {
    return this.request<ReliabilityDiagnostic>(
      `/diagnostics/${encodeURIComponent(runId)}/${encodeURIComponent(testCaseId)}`,
    );
  }

  // --- RAG ---
  async listRetrievalResults(params?: {
    runId?: string;
    testCaseId?: string;
    limit?: number;
    offset?: number;
  }): Promise<
    PaginatedResponse<{
      test_case_id: string;
      retrieved_chunks: unknown[];
      retrieval_evaluations: RetrievalEvaluationResult[];
    }>
  > {
    const q = new URLSearchParams();
    if (params?.runId) q.set("run_id", params.runId);
    if (params?.testCaseId) q.set("test_case_id", params.testCaseId);
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.offset) q.set("offset", String(params.offset));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return this.request<
      PaginatedResponse<{
        test_case_id: string;
        retrieved_chunks: unknown[];
        retrieval_evaluations: RetrievalEvaluationResult[];
      }>
    >(`/rag/retrieval-results${qs}`);
  }

  async getRetrievalResult(
    runId: string,
    testCaseId: string,
  ): Promise<{
    test_case_id: string;
    retrieved_chunks: unknown[];
    retrieval_evaluations: RetrievalEvaluationResult[];
  }> {
    return this.request<{
      test_case_id: string;
      retrieved_chunks: unknown[];
      retrieval_evaluations: RetrievalEvaluationResult[];
    }>(`/rag/retrieval-results/${encodeURIComponent(runId)}/${encodeURIComponent(testCaseId)}`);
  }

  // --- Robustness ---
  async getRunRobustness(runId: string): Promise<RunRobustnessBundle> {
    return this.request<RunRobustnessBundle>(`/robustness/${encodeURIComponent(runId)}`);
  }

  // --- Analytics ---
  async getAnalyticsOverview(): Promise<AnalyticsOverview> {
    return this.request<AnalyticsOverview>("/analytics/overview");
  }

  async getEvaluatorAnalytics(runId: string): Promise<{
    run_id: string;
    aggregates: Record<string, EvaluationAggregate>;
    agreements: EvaluatorAgreementSummary[];
  }> {
    return this.request<{
      run_id: string;
      aggregates: Record<string, EvaluationAggregate>;
      agreements: EvaluatorAgreementSummary[];
    }>(`/analytics/evaluators?run_id=${encodeURIComponent(runId)}`);
  }

  async getFailureAnalytics(runId: string): Promise<FailureDistribution> {
    return this.request<FailureDistribution>(
      `/analytics/failures?run_id=${encodeURIComponent(runId)}`,
    );
  }

  async getRetrievalAnalytics(runId: string): Promise<RetrievalSummary[]> {
    return this.request<RetrievalSummary[]>(
      `/analytics/retrieval?run_id=${encodeURIComponent(runId)}`,
    );
  }

  async getRobustnessAnalytics(runId: string): Promise<RobustnessSummary | null> {
    return this.request<RobustnessSummary | null>(
      `/analytics/robustness?run_id=${encodeURIComponent(runId)}`,
    );
  }

  async getPerformanceAnalytics(runId: string): Promise<ExecutionPerformanceSummary> {
    return this.request<ExecutionPerformanceSummary>(
      `/analytics/performance?run_id=${encodeURIComponent(runId)}`,
    );
  }

  async compareRuns(runIds: string[]): Promise<RunComparison> {
    const q = new URLSearchParams();
    for (const id of runIds) {
      q.append("run_ids", id);
    }
    return this.request<RunComparison>(`/analytics/comparisons?${q.toString()}`);
  }

  async getTrends(experimentId: string): Promise<TrendAnalysis> {
    return this.request<TrendAnalysis>(`/analytics/trends/${encodeURIComponent(experimentId)}`);
  }
}

export const defaultApiClient = new ApiClient();

