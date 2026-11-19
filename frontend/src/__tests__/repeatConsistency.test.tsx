// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import { ConnectionProvider } from "../context/ConnectionContext";
import { ApiClient } from "../api/client";
import { RepeatConsistency, formatOfTotal } from "../components/RepeatConsistency";
import type { RepeatConsistencyReport, RepeatGroupConsistency } from "../types/domain";

function group(overrides: Partial<RepeatGroupConsistency> = {}): RepeatGroupConsistency {
  return {
    experiment_id: "exp-a",
    experiment_name: "Experiment A",
    config_fingerprint: "fp",
    adapter_id: "ollama-rag-k3",
    adapter_recorded: true,
    num_runs: 2,
    run_ids: ["r1", "r2"],
    temperature: 0,
    model_id: "m",
    simulated_components: [],
    total_test_cases: 61,
    compared_cases: 60,
    excluded_cases: 1,
    identical_output_cases: 57,
    identical_output_rate: 0.95,
    output_changed_count: 3,
    output_changed_test_case_ids: ["q1", "q2", "q3"],
    case_pass_stable_cases: 58,
    case_pass_compared_cases: 60,
    case_pass_stability_rate: 58 / 60,
    case_pass_changed_count: 2,
    case_pass_changed_test_case_ids: ["q1", "q3"],
    evaluators: [
      {
        evaluator_name: "llm_judge",
        stable_cases: 59,
        compared_cases: 60,
        stability_rate: 59 / 60,
        excluded_cases: 0,
        excluded_missing: 0,
        excluded_not_success: 0,
        excluded_no_verdict: 0,
        unstable_count: 1,
        unstable_test_case_ids: ["q3"],
      },
      {
        evaluator_name: "llm_faithfulness_judge",
        stable_cases: 57,
        compared_cases: 57,
        stability_rate: 1,
        excluded_cases: 3,
        excluded_missing: 0,
        excluded_not_success: 1,
        excluded_no_verdict: 2,
        unstable_count: 0,
        unstable_test_case_ids: [],
      },
    ],
    ...overrides,
  };
}

function healthyClient(): ApiClient {
  const client = new ApiClient("http://localhost:8000/api/v1");
  vi.spyOn(client, "checkHealth").mockResolvedValue({
    status: "healthy",
    version: "0.1.0",
    environment: "test",
    timestamp: "2026-11-02T00:00:00Z",
    database: "local_filesystem",
    allowlisted_adapters: [],
  });
  return client;
}

describe("formatOfTotal", () => {
  it("shows numerator, denominator and rounded percent", () => {
    expect(formatOfTotal(57, 60)).toBe("57 of 60 (95%)");
    expect(formatOfTotal(0, 0)).toBe("0 of 0");
  });
});

describe("RepeatConsistency panel", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders a live group with denominators and exclusions", async () => {
    const client = healthyClient();
    const report: RepeatConsistencyReport = { experiment_id: null, groups: [group()] };
    const spy = vi.spyOn(client, "getRepeatConsistency").mockResolvedValue(report);
    render(
      <ConnectionProvider client={client}>
        <RepeatConsistency />
      </ConnectionProvider>,
    );

    expect(await screen.findByText("Experiment A")).toBeTruthy();
    expect(spy).toHaveBeenCalledWith(undefined);
    expect(screen.getByText("Do repeated runs agree?")).toBeTruthy();
    expect(screen.getByText("57 of 60 (95%)")).toBeTruthy();
    expect(screen.getByText("58 of 60 (97%)")).toBeTruthy();
    expect(screen.getByText("59 of 60 (98%)")).toBeTruthy();
    expect(screen.getByText("57 of 57 (100%)")).toBeTruthy();
    expect(screen.getByText(/2 runs · temperature 0/)).toBeTruthy();
    expect(screen.getByText(/1 of 61 questions left out/)).toBeTruthy();
    expect(screen.getByText(/3 not comparable, left out: 1 skipped or errored in a run, 2 gave no verdict/)).toBeTruthy();
    // A real group carries no simulated label.
    expect(screen.queryByText("Simulated")).toBeNull();
  });

  it("labels simulated groups and unrecorded adapters", async () => {
    const client = healthyClient();
    vi.spyOn(client, "getRepeatConsistency").mockResolvedValue({
      experiment_id: null,
      groups: [
        group({
          simulated_components: ["generation: deterministic mock adapter (canned answers)"],
          adapter_id: null,
          adapter_recorded: false,
        }),
      ],
    });
    render(
      <ConnectionProvider client={client}>
        <RepeatConsistency />
      </ConnectionProvider>,
    );

    expect(await screen.findByText("Simulated")).toBeTruthy();
    expect(screen.getByText("Adapter not recorded")).toBeTruthy();
    expect(screen.getByText("generation: deterministic mock adapter (canned answers)")).toBeTruthy();
  });

  it("explains how to get repeats when the live API has none", async () => {
    const client = healthyClient();
    vi.spyOn(client, "getRepeatConsistency").mockResolvedValue({ experiment_id: null, groups: [] });
    render(
      <ConnectionProvider client={client}>
        <RepeatConsistency />
      </ConnectionProvider>,
    );

    expect(await screen.findByText("No repeated runs yet.")).toBeTruthy();
    expect(screen.getByText(/Compare models \/ repeat/)).toBeTruthy();
  });

  it("shows an empty state in demo mode instead of numbers, without calling the API", async () => {
    localStorage.setItem("llm_reliability_data_mode", "fixture");
    const client = healthyClient();
    const spy = vi.spyOn(client, "getRepeatConsistency");
    render(
      <ConnectionProvider client={client}>
        <RepeatConsistency />
      </ConnectionProvider>,
    );

    expect(screen.getByText("Repeat analysis needs the live API.")).toBeTruthy();
    expect(screen.queryByText(/ of \d+ \(/)).toBeNull();
    await waitFor(() => expect(spy).not.toHaveBeenCalled());
  });

  it("is mounted on the Analysis reliability tab", async () => {
    localStorage.setItem("llm_reliability_data_mode", "fixture");
    const client = healthyClient();
    render(
      <ConnectionProvider client={client}>
        <MemoryRouter initialEntries={["/analysis?tab=reliability"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    expect(await screen.findByRole("heading", { name: "Do repeated runs agree?" })).toBeTruthy();
    expect(screen.getByText("Repeat analysis needs the live API.")).toBeTruthy();
  });
});
