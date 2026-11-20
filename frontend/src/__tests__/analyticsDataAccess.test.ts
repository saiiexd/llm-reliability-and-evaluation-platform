import { describe, expect, it } from "vitest";
import {
  getAnalyticsOverview,
  getComparison,
  getEvaluatorAnalytics,
  getFailureAnalytics,
  getPerformanceAnalytics,
  getRetrievalAnalytics,
  getRobustnessAnalytics,
  getTrendForExperiment,
  listAllRuns,
  listComparisons,
  listExperiments,
} from "../data/dataAccess";

describe("Analytics Data Access Layer (Prompt 6)", () => {
  it("loads platform-wide analytics overview preserving counts and coverage", () => {
    const overview = getAnalyticsOverview();
    expect(overview).toBeDefined();
    expect(overview.total_experiments).toBeGreaterThan(0);
    expect(overview.total_runs).toBeGreaterThan(0);
    expect(overview.total_test_cases).toBeGreaterThan(0);
    expect(overview.total_evaluations).toBeGreaterThan(0);
    expect(typeof overview.evaluator_coverage).toBe("object");
    expect(typeof overview.primary_failure_distribution).toBe("object");
    expect(overview.recent_runs.length).toBeGreaterThan(0);
  });

  it("retrieves comparative studies with explicit compatibility levels", () => {
    const comparisons = listComparisons();
    expect(comparisons).toBeDefined();

    const identicalComp = getComparison("identical");
    expect(identicalComp).toBeDefined();
    expect(identicalComp?.compatibility).toBe("identical_configuration");
    expect(identicalComp?.source_run_ids.length).toBe(2);

    const interventionComp = getComparison("intervention");
    expect(interventionComp).toBeDefined();
    expect(interventionComp?.compatibility).toBe("partially_different_configuration");

    const incompatibleComp = getComparison("incompatible");
    expect(incompatibleComp).toBeDefined();
    expect(incompatibleComp?.compatibility).toBe("materially_incompatible");
    expect(incompatibleComp?.compatibility_reasons.length).toBeGreaterThan(0);
  });

  it("retrieves evaluator analytics and pairwise agreements per run", () => {
    const runs = listAllRuns();
    const firstRun = runs[0];
    const analytics = getEvaluatorAnalytics(firstRun.run_id);

    expect(analytics).toBeDefined();
    expect(analytics?.aggregates).toBeDefined();
    const aggs = Object.values(analytics?.aggregates ?? {});
    expect(aggs.length).toBeGreaterThan(0);

    const firstAgg = aggs[0];
    expect(firstAgg.evaluator_name).toBeDefined();
    expect(firstAgg.pass_rate_numerator).toBeDefined();
    expect(firstAgg.pass_rate_denominator).toBeDefined();
  });

  it("retrieves failure analytics with independent secondary findings", () => {
    const runs = listAllRuns();
    const firstRun = runs[0];
    const failureDist = getFailureAnalytics(firstRun.run_id);

    expect(failureDist).toBeDefined();
    expect(failureDist?.total_cases).toBeGreaterThan(0);
    expect(typeof failureDist?.primary_categories).toBe("object");
    expect(typeof failureDist?.secondary_findings).toBe("object");
  });

  it("retrieves retrieval analytics preserving metric names and K values", () => {
    const runs = listAllRuns();
    const ragRun = runs.find((r) => getRetrievalAnalytics(r.run_id).length > 0);
    expect(ragRun).toBeDefined();

    if (ragRun) {
      const summaries = getRetrievalAnalytics(ragRun.run_id);
      expect(summaries.length).toBeGreaterThan(0);
      const hit1 = summaries.find((s) => s.metric_name === "hit_at_k" && s.k === 1);
      expect(hit1).toBeDefined();
      expect(hit1?.evaluated_cases).toBeGreaterThan(0);
    }
  });

  it("retrieves robustness summary when instrumented", () => {
    const runs = listAllRuns();
    const robRun = runs.find((r) => getRobustnessAnalytics(r.run_id) !== undefined);
    expect(robRun).toBeDefined();

    if (robRun) {
      const robSummary = getRobustnessAnalytics(robRun.run_id);
      expect(robSummary).toBeDefined();
      expect(robSummary?.total_pairs).toBeGreaterThan(0);
      expect(typeof robSummary?.consistency_rate).toBe("number");
    }
  });

  it("retrieves performance telemetry reporting tokens and costs as unavailable", () => {
    const runs = listAllRuns();
    const firstRun = runs[0];
    const perf = getPerformanceAnalytics(firstRun.run_id);

    expect(perf).toBeDefined();
    expect(perf?.tokens_available).toBe(false);
    expect(perf?.cost_available).toBe(false);
    expect(perf?.notes.length).toBeGreaterThan(0);
    expect(perf?.notes.some((n) => n.toLowerCase().includes("cost"))).toBe(true);
  });

  it("retrieves longitudinal trend analysis for multi-run experiments", () => {
    const experiments = listExperiments();
    const expId = experiments[0].experiment_id;
    const trend = getTrendForExperiment(expId);

    expect(trend).toBeDefined();
    expect(trend?.run_ids.length).toBeGreaterThanOrEqual(2);
    expect(trend?.has_trend).toBe(true);
    expect(trend?.timestamps.length).toBe(trend?.run_ids.length);
  });
});

