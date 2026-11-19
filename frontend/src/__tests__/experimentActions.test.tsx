// @vitest-environment jsdom
/**
 * The redesign rewrote the create form and the run controls. These tests pin
 * the requests they send to the backend so a UI change can't silently alter
 * them.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import { ConnectionProvider } from "../context/ConnectionContext";
import { ApiClient } from "../api/client";
import { NewExperimentForm } from "../components/NewExperimentForm";
import type { ExperimentDefinition } from "../types/domain";

describe("experiment actions send unchanged requests", () => {
  let client: ApiClient;

  beforeEach(() => {
    localStorage.clear();
    client = new ApiClient("http://localhost:8000/api/v1");
    vi.spyOn(client, "checkHealth").mockResolvedValue({
      status: "healthy",
      version: "0.1.0",
      environment: "test",
      timestamp: "2026-11-02T00:00:00Z",
      database: "local_filesystem",
      allowlisted_adapters: ["mock-adapter-v1"],
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("create form posts the same payload shape as before the redesign", async () => {
    const create = vi.spyOn(client, "createExperiment").mockResolvedValue({} as ExperimentDefinition);
    const onCreated = vi.fn();
    render(
      <ConnectionProvider client={client}>
        <NewExperimentForm onCreated={onCreated} onCancel={() => {}} />
      </ConnectionProvider>,
    );

    fireEvent.change(screen.getByPlaceholderText("e.g. Geography baseline"), { target: { value: "Geo test" } });
    fireEvent.click(screen.getByRole("button", { name: /Faithfulness \(LLM judge\)/ }));
    fireEvent.click(screen.getByRole("button", { name: "Save experiment" }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("geo-test"));
    expect(create).toHaveBeenCalledTimes(1);
    const payload = create.mock.calls[0][0] as Record<string, unknown>;
    expect(payload).toMatchObject({
      experiment_id: "geo-test",
      name: "Geo test",
      description: undefined,
      dataset_name: "qa-geography-baseline-v1",
      model_config: { model_id: "mock-adapter-v1", temperature: 0, max_tokens: 100 },
      evaluator_names: ["exact_match", "llm_faithfulness_judge"],
      metadata: {},
    });
    expect(payload.test_cases).toHaveLength(5);
  });

  it("Run button executes the experiment with the selected adapter", async () => {
    vi.spyOn(client, "getExperiment").mockResolvedValue({
      schema_version: 1,
      experiment_id: "exp-x",
      name: "Experiment X",
      description: null,
      dataset_version: { dataset_name: "d", version_id: "v", test_case_count: 2 },
      model_config: { model_id: "mock-adapter-v1", temperature: null, max_tokens: null, extra_params: {} },
      evaluator_configs: [{ evaluator_name: "exact_match", parameters: {} }],
      metadata: {},
      created_at: "2026-11-02T00:00:00Z",
      config_fingerprint: "abc",
    });
    vi.spyOn(client, "listRuns").mockResolvedValue({ items: [], pagination: {} } as never);
    vi.spyOn(client, "listProviders").mockResolvedValue({ providers: [] } as never);
    const execute = vi.spyOn(client, "executeExperiment").mockResolvedValue({
      run_id: "run-1",
      duration_seconds: 0.1,
    } as never);

    render(
      <ConnectionProvider client={client}>
        <MemoryRouter initialEntries={["/experiments/exp-x"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(1));
    // Non-RAG experiment (no extra_params.rag) defaults to the plain mock
    // adapter; a RAG pipeline would not match its recorded configuration.
    expect(execute).toHaveBeenCalledWith("exp-x", {
      adapter_id: "mock-adapter-v1",
      allow_external_inference: false,
      model_config: undefined,
    });
    expect(await screen.findByRole("link", { name: /See the results/ })).toBeDefined();
  });
});
