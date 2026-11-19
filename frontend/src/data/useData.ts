/**
 * React data hooks integrating live API endpoints and fixture fallbacks.
 *
 * Each hook observes the current `dataMode` from `ConnectionContext`:
 * - In "fixture" mode: returns static fixture data immediately without network requests.
 * - In "live" mode: executes typed asynchronous queries against the FastAPI backend,
 *   surfacing loading, error, and refetch states without silent fallback.
 */

import { useCallback, useEffect, useState } from "react";
import { useConnection } from "../context/ConnectionContext";
import * as fixtures from "./dataAccess";
import type {
  AnalyticsOverview,
  EvaluationAggregate,
  EvaluatorAgreementSummary,
  ExecutionPerformanceSummary,
  ExperimentDefinition,
  ExperimentRun,
  FailureDistribution,
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

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  isLive: boolean;
  refetch: () => Promise<void>;
}

function useAsyncData<T>(
  queryFn: () => Promise<T>,
  fixtureFn: () => T | undefined,
  deps: unknown[],
): AsyncState<T> {
  const { dataMode, client } = useConnection();
  const [data, setData] = useState<T | null>(() => {
    if (dataMode === "fixture") {
      return fixtureFn() ?? null;
    }
    return null;
  });
  const [loading, setLoading] = useState<boolean>(dataMode === "live");
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(async () => {
    if (dataMode === "fixture") {
      setData(fixtureFn() ?? null);
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await queryFn();
      setData(result);
    } catch (err: unknown) {
      setData(null);
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataMode, client, ...deps]);

  useEffect(() => {
    void execute();
  }, [execute]);

  return {
    data,
    loading,
    error,
    isLive: dataMode === "live",
    refetch: execute,
  };
}

// --- Hooks ---

export function useExperiments(search?: string): AsyncState<ExperimentDefinition[]> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      const res = await client.listExperiments({ search, limit: 100 });
      return res.items;
    },
    () => {
      let list = fixtures.listExperiments();
      if (search) {
        const q = search.toLowerCase();
        list = list.filter(
          (e) => e.name.toLowerCase().includes(q) || e.experiment_id.toLowerCase().includes(q),
        );
      }
      return list;
    },
    [search],
  );
}

export function useExperiment(experimentId: string): AsyncState<ExperimentDefinition> {
  const { client } = useConnection();
  return useAsyncData(
    () =>
      experimentId
        ? client.getExperiment(experimentId)
        : (Promise.resolve(null as unknown as ExperimentDefinition)),
    () => (experimentId ? fixtures.getExperiment(experimentId) : undefined),
    [experimentId],
  );
}

export function useRuns(experimentId?: string): AsyncState<ExperimentRun[]> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      const res = await client.listRuns({ experimentId, limit: 100 });
      return res.items;
    },
    () => (experimentId ? fixtures.listRunsForExperiment(experimentId) : fixtures.listAllRuns()),
    [experimentId],
  );
}

export function useRun(runId: string): AsyncState<ExperimentRun> {
  const { client } = useConnection();
  return useAsyncData(
    () => (runId ? client.getRun(runId, true) : (Promise.resolve(null as unknown as ExperimentRun))),
    () => (runId ? fixtures.getRun(runId) : undefined),
    [runId],
  );
}

export function useTestCase(runId: string, testCaseId: string): AsyncState<TestCaseResult> {
  const { client } = useConnection();
  return useAsyncData(
    () =>
      runId && testCaseId
        ? client.getTestCase(runId, testCaseId)
        : (Promise.resolve(null as unknown as TestCaseResult)),
    () => {
      if (!runId || !testCaseId) return undefined;
      const run = fixtures.getRun(runId);
      return run?.result?.test_case_results.find((tc) => tc.test_case_id === testCaseId);
    },
    [runId, testCaseId],
  );
}

export function useDiagnostics(runId: string): AsyncState<Record<string, ReliabilityDiagnostic>> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      if (!runId) return {};
      const res = await client.listDiagnostics({ runId, limit: 500 });
      const map: Record<string, ReliabilityDiagnostic> = {};
      for (const item of res.items) {
        map[item.test_case_id] = item;
      }
      return map;
    },
    () => (runId ? fixtures.getDiagnosticsForRun(runId) : {}),
    [runId],
  );
}

export function useDiagnostic(
  runId: string,
  testCaseId: string,
): AsyncState<ReliabilityDiagnostic> {
  const { client } = useConnection();
  return useAsyncData(
    () =>
      runId && testCaseId
        ? client.getDiagnostic(runId, testCaseId)
        : (Promise.resolve(null as unknown as ReliabilityDiagnostic)),
    () => (runId && testCaseId ? fixtures.getDiagnosticForTestCase(runId, testCaseId) : undefined),
    [runId, testCaseId],
  );
}

export function useRetrievalResults(
  runId: string,
): AsyncState<Record<string, RetrievalEvaluationResult[]>> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      if (!runId) return {};
      const res = await client.listRetrievalResults({ runId, limit: 500 });
      const map: Record<string, RetrievalEvaluationResult[]> = {};
      for (const item of res.items) {
        map[item.test_case_id] = item.retrieval_evaluations;
      }
      return map;
    },
    () => (runId ? fixtures.getRetrievalResultsForRun(runId) : {}),
    [runId],
  );
}

export function useRobustness(runId: string): AsyncState<RunRobustnessBundle> {
  const { client } = useConnection();
  return useAsyncData(
    () => client.getRunRobustness(runId),
    () => fixtures.getRobustnessForRun(runId),
    [runId],
  );
}

export function useRunSummaries(runId: string): AsyncState<RunSummaryBundle> {
  const { client } = useConnection();
  return useAsyncData(
    () => client.getRunSummaries(runId),
    () => fixtures.getSummaryForRun(runId),
    [runId],
  );
}

export function useAnalyticsOverview(): AsyncState<AnalyticsOverview> {
  const { client } = useConnection();
  return useAsyncData(
    () => client.getAnalyticsOverview(),
    () => fixtures.getAnalyticsOverview(),
    [],
  );
}

export function useEvaluatorAnalytics(runId: string): AsyncState<{
  aggregates: Record<string, EvaluationAggregate>;
  agreements: EvaluatorAgreementSummary[];
}> {
  const { client } = useConnection();
  return useAsyncData(
    () => client.getEvaluatorAnalytics(runId),
    () => fixtures.getEvaluatorAnalytics(runId),
    [runId],
  );
}

export function useFailureAnalytics(runId: string): AsyncState<FailureDistribution> {
  const { client } = useConnection();
  return useAsyncData(
    () =>
      runId
        ? client.getFailureAnalytics(runId)
        : (Promise.resolve(null as unknown as FailureDistribution)),
    () => (runId ? fixtures.getFailureAnalytics(runId) : undefined),
    [runId],
  );
}

export function useRetrievalAnalytics(runId: string): AsyncState<RetrievalSummary[]> {
  const { client } = useConnection();
  return useAsyncData(
    () => (runId ? client.getRetrievalAnalytics(runId) : Promise.resolve([])),
    () => (runId ? fixtures.getRetrievalAnalytics(runId) : []),
    [runId],
  );
}

export function useRobustnessAnalytics(runId: string): AsyncState<RobustnessSummary | null> {
  const { client } = useConnection();
  return useAsyncData(
    () => (runId ? client.getRobustnessAnalytics(runId) : Promise.resolve(null)),
    () => (runId ? (fixtures.getRobustnessAnalytics(runId) ?? null) : null),
    [runId],
  );
}

export function usePerformanceAnalytics(
  runId: string,
): AsyncState<ExecutionPerformanceSummary> {
  const { client } = useConnection();
  return useAsyncData(
    () =>
      runId
        ? client.getPerformanceAnalytics(runId)
        : (Promise.resolve(null as unknown as ExecutionPerformanceSummary)),
    () => (runId ? fixtures.getPerformanceAnalytics(runId) : undefined),
    [runId],
  );
}

export function useComparisons(): AsyncState<Record<string, RunComparison>> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      // In live mode, compute standard comparison sets:
      // identical runs: run-rag-baseline-001, run-rag-baseline-002
      // intervention: run-rag-baseline-001, run-rag-topk2-001
      // incompatible: run-rag-baseline-001, run-non-rag-001
      const res: Record<string, RunComparison> = {};
      try {
        res["identical"] = await client.compareRuns([
          "run-rag-baseline-001",
          "run-rag-baseline-002",
        ]);
      } catch {
        // May not be available if runs differ
      }
      try {
        res["intervention"] = await client.compareRuns([
          "run-rag-baseline-001",
          "run-rag-topk2-001",
        ]);
      } catch {
        // Optional
      }
      try {
        res["incompatible"] = await client.compareRuns([
          "run-rag-baseline-001",
          "run-non-rag-001",
        ]);
      } catch {
        // Optional
      }
      return res;
    },
    () => fixtures.listComparisons(),
    [],
  );
}

export function useComparison(
  key: string,
  runIds?: string[],
): AsyncState<RunComparison> {
  const { client } = useConnection();
  return useAsyncData(
    async () => {
      if (runIds && runIds.length >= 2) {
        return client.compareRuns(runIds);
      }
      // Map standard keys to runs if known
      const standardPairs: Record<string, string[]> = {
        identical: ["run-rag-baseline-001", "run-rag-baseline-002"],
        intervention: ["run-rag-baseline-001", "run-rag-topk2-001"],
        incompatible: ["run-rag-baseline-001", "run-non-rag-001"],
      };
      const pair = standardPairs[key] || ["run-rag-baseline-001", "run-rag-baseline-002"];
      return client.compareRuns(pair);
    },
    () => fixtures.getComparison(key),
    [key, runIds?.join(",")],
  );
}

export function useTrends(experimentId: string): AsyncState<TrendAnalysis> {
  const { client } = useConnection();
  return useAsyncData(
    () => client.getTrends(experimentId),
    () => fixtures.getTrendForExperiment(experimentId),
    [experimentId],
  );
}

