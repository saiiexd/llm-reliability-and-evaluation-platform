import { describe, expect, it } from "vitest";
import {
  fixtureMeta,
  getDiagnosticForTestCase,
  getDiagnosticsForRun,
  getExperiment,
  getRetrievalResultsForRun,
  getRobustnessForRun,
  getRun,
  getSummaryForRun,
  listAllRuns,
  listExperiments,
  listRunsForExperiment,
} from "../data/dataAccess";

describe("Typed Data Access Boundary", () => {
  it("marks all fixture data explicitly with is_fixture_data", () => {
    expect(fixtureMeta.is_fixture_data).toBe(true);
    expect(fixtureMeta.generated_by).toContain("scripts/export_frontend_fixtures.py");
  });

  it("lists all loaded experiments", () => {
    const experiments = listExperiments();
    expect(Array.isArray(experiments)).toBe(true);
    expect(experiments.length).toBeGreaterThan(0);
    const first = experiments[0];
    expect(first.experiment_id).toBeDefined();
    expect(first.config_fingerprint).toBeDefined();
  });

  it("finds experiment by id", () => {
    const experiments = listExperiments();
    const first = experiments[0];
    const found = getExperiment(first.experiment_id);
    expect(found).toEqual(first);
    expect(getExperiment("nonexistent-exp-id")).toBeUndefined();
  });

  it("lists runs for an experiment and all runs", () => {
    const runs = listAllRuns();
    expect(runs.length).toBeGreaterThan(0);
    const firstRun = runs[0];
    expect(firstRun.run_id).toBeDefined();

    const expRuns = listRunsForExperiment(firstRun.experiment_id);
    expect(expRuns.some((r) => r.run_id === firstRun.run_id)).toBe(true);
  });

  it("loads run by id", () => {
    const runs = listAllRuns();
    const first = runs[0];
    const found = getRun(first.run_id);
    expect(found).toBeDefined();
    expect(found?.run_id).toBe(first.run_id);
  });

  it("retrieves diagnostics for a run", () => {
    const runs = listAllRuns();
    const first = runs[0];
    const diags = getDiagnosticsForRun(first.run_id);
    expect(typeof diags).toBe("object");

    const tcIds = Object.keys(diags);
    if (tcIds.length > 0) {
      const tcDiag = getDiagnosticForTestCase(first.run_id, tcIds[0]);
      expect(tcDiag).toBeDefined();
      expect(tcDiag?.test_case_id).toBe(tcIds[0]);
    }
  });

  it("retrieves summaries, robustness, and retrieval metrics", () => {
    const runs = listAllRuns();
    const first = runs[0];

    const summary = getSummaryForRun(first.run_id);
    expect(summary).toBeDefined();

    const retrievalResults = getRetrievalResultsForRun(first.run_id);
    expect(typeof retrievalResults).toBe("object");

    const runWithRobustness = runs.find((r) => getRobustnessForRun(r.run_id) !== undefined);
    expect(runWithRobustness).toBeDefined();
    if (runWithRobustness) {
      const robustness = getRobustnessForRun(runWithRobustness.run_id);
      expect(robustness).toBeDefined();
      expect(robustness?.consistency.num_comparisons).toBeGreaterThan(0);
    }
  });
});

