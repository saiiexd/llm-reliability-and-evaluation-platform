import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";

describe("ApiClient", () => {
  let client: ApiClient;

  beforeEach(() => {
    client = new ApiClient("http://test-server:8000/api/v1", 2000);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("normalizes base URL by stripping trailing slashes", () => {
    const c = new ApiClient("http://localhost:8000/api/v1///");
    expect(c.getBaseUrl()).toBe("http://localhost:8000/api/v1");
  });

  it("executes successful GET /health", async () => {
    const mockHealth = {
      status: "healthy",
      version: "0.1.0",
      environment: "test",
      timestamp: "2026-11-02T00:00:00Z",
      database: "local_filesystem",
      allowlisted_adapters: ["mock-adapter-v1"],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ "x-request-id": "req-123" }),
      json: async () => mockHealth,
    });

    const result = await client.checkHealth();
    expect(result.status).toBe("healthy");
    expect(result.version).toBe("0.1.0");
    expect(global.fetch).toHaveBeenCalledWith(
      "http://test-server:8000/api/v1/health",
      expect.objectContaining({
        headers: expect.objectContaining({ Accept: "application/json" }),
      }),
    );
  });

  it("handles structured 404 API error response", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
      headers: new Headers({ "x-request-id": "req-404" }),
      json: async () => ({
        error: {
          code: "RESOURCE_NOT_FOUND",
          message: "Experiment 'exp-999' not found.",
          details: [],
        },
      }),
    });

    try {
      await client.getExperiment("exp-999");
      expect.fail("Should have thrown ApiError");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(404);
      expect(apiErr.code).toBe("RESOURCE_NOT_FOUND");
      expect(apiErr.message).toBe("Experiment 'exp-999' not found.");
      expect(apiErr.requestId).toBe("req-404");
    }
  });

  it("handles network failure without silent fallback", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Failed to fetch"));

    try {
      await client.listExperiments();
      expect.fail("Should have thrown ApiError");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.code).toBe("CONNECTION_ERROR");
      expect(apiErr.message).toContain("Failed to fetch");
    }
  });

  it("executes experiment execution POST with payload", async () => {
    const mockRunResponse = {
      run_id: "run-new-001",
      experiment_id: "exp-001",
      status: "COMPLETED",
      duration_seconds: 1.25,
      test_cases_count: 5,
      num_succeeded: 5,
      num_failed: 0,
      created_at: "2026-11-02T00:00:00Z",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ "x-request-id": "req-exec" }),
      json: async () => mockRunResponse,
    });

    const res = await client.executeExperiment("exp-001", {
      adapter_id: "mock-adapter-v1",
    });

    expect(res.run_id).toBe("run-new-001");
    expect(res.status).toBe("COMPLETED");
    expect(global.fetch).toHaveBeenCalledWith(
      "http://test-server:8000/api/v1/experiments/exp-001/execute",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ adapter_id: "mock-adapter-v1" }),
      }),
    );
  });

  it("fetches providers readiness list", async () => {
    const mockProviders = {
      providers: [
        {
          provider_id: "openai",
          name: "OpenAI",
          is_configured: true,
          base_url: "https://api.openai.com/v1",
          requires_api_key: true,
          supported_models: [
            {
              model_id: "gpt-4o-mini",
              display_name: "GPT-4o Mini",
              context_window: 128000,
              default_temperature: 0.0,
            },
          ],
          default_model: "gpt-4o-mini",
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => mockProviders,
    });

    const res = await client.listProviders();
    expect(res.providers).toHaveLength(1);
    expect(res.providers[0].provider_id).toBe("openai");
    expect(res.providers[0].is_configured).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://test-server:8000/api/v1/providers",
      expect.anything(),
    );
  });

  it("executes benchmark with multiple adapters and repetitions", async () => {
    const mockBenchResponse = {
      benchmark_id: "bench-test-123",
      experiment_id: "exp-001",
      total_runs: 2,
      runs: [
        {
          run_id: "bench-test-123-mock-rep1",
          experiment_id: "exp-001",
          status: "succeeded",
          duration_seconds: 0.5,
          num_succeeded: 2,
          num_failed: 0,
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      headers: new Headers(),
      json: async () => mockBenchResponse,
    });

    const res = await client.executeBenchmark({
      experiment_id: "exp-001",
      adapter_ids: ["mock-adapter-v1"],
      repetitions: 1,
      allow_external_inference: true,
    });

    expect(res.benchmark_id).toBe("bench-test-123");
    expect(res.total_runs).toBe(2);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://test-server:8000/api/v1/benchmarks/execute",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          experiment_id: "exp-001",
          adapter_ids: ["mock-adapter-v1"],
          repetitions: 1,
          allow_external_inference: true,
        }),
      }),
    );
  });

  it("fetches repeated-run consistency, with and without an experiment filter", async () => {
    const body = {
      experiment_id: "exp 1",
      groups: [{ experiment_id: "exp 1", identical_output_cases: 57, compared_cases: 60, evaluators: [] }],
    };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => body,
    });

    const res = await client.getRepeatConsistency("exp 1");
    expect(res.groups[0].identical_output_cases).toBe(57);
    expect(res.groups[0].compared_cases).toBe(60);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://test-server:8000/api/v1/analytics/consistency?experiment_id=exp%201",
      expect.anything(),
    );

    await client.getRepeatConsistency();
    expect(global.fetch).toHaveBeenLastCalledWith(
      "http://test-server:8000/api/v1/analytics/consistency",
      expect.anything(),
    );
  });
});

