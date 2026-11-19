"""
Analytics query and service layer.

Provides unified analytical access to experiment runs, cross-run comparisons,
evaluator agreement calculations, failure category distributions, retrieval
aggregates, robustness metrics, and longitudinal trends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_reliability.analytics.aggregations import (
    aggregate_evaluation,
    aggregate_failures,
    aggregate_retrieval,
    aggregate_robustness,
    measure_performance,
)
from llm_reliability.analytics.agreement import compute_all_pairwise_agreements
from llm_reliability.analytics.comparison import compare_runs
from llm_reliability.analytics.models import (
    AnalyticsOverview,
    EvaluationAggregate,
    EvaluatorAgreementSummary,
    ExecutionPerformanceSummary,
    FailureDistribution,
    RetrievalSummary,
    RobustnessSummary,
    RunComparison,
    TrendAnalysis,
)
from llm_reliability.analytics.trends import compute_trend_analysis
from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
from llm_reliability.diagnostics.robustness import RobustnessResult
from llm_reliability.experiments.models import ExperimentRun
from llm_reliability.experiments.repository import ExperimentRepository
from llm_reliability.rag.evaluators import RetrievalEvaluationResult


class AnalyticsService:
    """Analytical service operating across experiments, runs, and derived evaluation artifacts."""

    def __init__(
        self,
        repository: ExperimentRepository,
        diagnostics_loader: Callable[[str], list[ReliabilityDiagnostic]] | None = None,
        retrieval_loader: Callable[[str], list[RetrievalEvaluationResult]] | None = None,
        robustness_loader: Callable[[str], list[RobustnessResult]] | None = None,
    ) -> None:
        self.repository = repository
        self.diagnostics_loader = diagnostics_loader
        self.retrieval_loader = retrieval_loader
        self.robustness_loader = robustness_loader

    def get_run(self, run_id: str) -> ExperimentRun:
        """Retrieve an experiment run from the underlying repository."""
        return self.repository.load_run(run_id)

    def get_overview(self) -> AnalyticsOverview:
        """Compute platform-wide aggregate analytics across all stored experiments and runs."""
        experiment_ids = self.repository.list_experiment_ids()
        total_experiments = len(experiment_ids)

        all_run_ids = self.repository.list_run_ids()
        total_runs = len(all_run_ids)

        total_test_cases = 0
        total_evaluations = 0
        evaluator_coverage: dict[str, int] = {}
        disagreement_count = 0
        primary_failures: dict[str, int] = {}
        retrieval_runs_count = 0
        mrr_scores: list[float] = []
        recent_runs: list[dict[str, Any]] = []

        for r_id in all_run_ids:
            run = self.repository.load_run(r_id)
            tc_count = len(run.result.test_case_results) if run.result else 0
            total_test_cases += tc_count

            # Aggregates
            agg = aggregate_evaluation(run)
            for eval_name, ea in agg.items():
                evaluator_coverage[eval_name] = (
                    evaluator_coverage.get(eval_name, 0) + ea.num_evaluated
                )
                total_evaluations += ea.num_evaluated

            # Evaluator agreements
            agreements = compute_all_pairwise_agreements(run)
            for a in agreements:
                disagreement_count += a.disagreements

            # Diagnostics
            diags: list[ReliabilityDiagnostic] = []
            if self.diagnostics_loader:
                try:
                    diags = self.diagnostics_loader(r_id)
                except Exception:
                    diags = []
            if diags:
                f_dist = aggregate_failures(diags, r_id)
                for c in f_dist.primary_categories.values():
                    primary_failures[c.category.value] = (
                        primary_failures.get(c.category.value, 0) + c.count
                    )

            # Retrieval
            ret_results: list[RetrievalEvaluationResult] = []
            if self.retrieval_loader:
                try:
                    ret_results = self.retrieval_loader(r_id)
                except Exception:
                    ret_results = []
            if ret_results:
                retrieval_runs_count += 1
                r_sums = aggregate_retrieval(ret_results)
                for rs in r_sums:
                    if rs.metric_name == "mrr" and rs.mean_score is not None:
                        mrr_scores.append(rs.mean_score)

            recent_runs.append(
                {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "started_at": run.started_at,
                    "status": run.status.value,
                    "case_count": tc_count,
                    "config_fingerprint": run.config_fingerprint,
                }
            )

        # Sort recent runs newest first
        recent_runs.sort(key=lambda r: str(r["started_at"]), reverse=True)

        mean_mrr = (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else None

        return AnalyticsOverview(
            total_experiments=total_experiments,
            total_runs=total_runs,
            total_test_cases=total_test_cases,
            total_evaluations=total_evaluations,
            evaluator_coverage=evaluator_coverage,
            disagreement_count=disagreement_count,
            primary_failure_distribution=primary_failures,
            retrieval_runs_count=retrieval_runs_count,
            mean_retrieval_mrr=mean_mrr,
            recent_runs=recent_runs[:10],
        )

    def get_run_evaluation_aggregates(self, run_id: str) -> dict[str, EvaluationAggregate]:
        """Aggregate evaluation metrics by evaluator for a specific run."""
        run = self.get_run(run_id)
        return aggregate_evaluation(run)

    def get_run_failures(
        self,
        run_id: str,
        diagnostics: list[ReliabilityDiagnostic] | None = None,
    ) -> FailureDistribution:
        """Compute primary and secondary failure category distributions for a run."""
        if diagnostics is None and self.diagnostics_loader:
            diagnostics = self.diagnostics_loader(run_id)
        return aggregate_failures(diagnostics or [], run_id)

    def get_run_retrieval(
        self,
        run_id: str,
        retrieval_results: list[RetrievalEvaluationResult] | None = None,
    ) -> list[RetrievalSummary]:
        """Compute retrieval metrics (Hit@K, Recall@K, MRR) for a run."""
        if retrieval_results is None and self.retrieval_loader:
            retrieval_results = self.retrieval_loader(run_id)
        return aggregate_retrieval(retrieval_results or [])

    def get_run_agreements(self, run_id: str) -> list[EvaluatorAgreementSummary]:
        """Compute pairwise evaluator concordance for a run."""
        run = self.get_run(run_id)
        return compute_all_pairwise_agreements(run)

    def get_run_robustness(
        self,
        run_id: str,
        robustness_results: list[RobustnessResult] | None = None,
    ) -> RobustnessSummary | None:
        """Compute robustness consistency across input perturbations for a run."""
        if robustness_results is None and self.robustness_loader:
            robustness_results = self.robustness_loader(run_id)
        if not robustness_results:
            return None
        return aggregate_robustness(robustness_results)

    def get_run_performance(self, run_id: str) -> ExecutionPerformanceSummary:
        """Extract execution durations and latency telemetry for a run."""
        run = self.get_run(run_id)
        return measure_performance(run)

    def compare(
        self,
        run_ids: list[str],
        diagnostics_by_run: dict[str, list[ReliabilityDiagnostic]] | None = None,
        retrieval_by_run: dict[str, list[RetrievalEvaluationResult]] | None = None,
        robustness_by_run: dict[str, list[RobustnessResult]] | None = None,
    ) -> RunComparison:
        """Perform side-by-side comparative analysis across two or more runs."""
        runs = [self.get_run(r_id) for r_id in run_ids]

        if diagnostics_by_run is None and self.diagnostics_loader:
            diagnostics_by_run = {}
            for r_id in run_ids:
                try:
                    diagnostics_by_run[r_id] = self.diagnostics_loader(r_id)
                except Exception:
                    diagnostics_by_run[r_id] = []

        if retrieval_by_run is None and self.retrieval_loader:
            retrieval_by_run = {}
            for r_id in run_ids:
                try:
                    retrieval_by_run[r_id] = self.retrieval_loader(r_id)
                except Exception:
                    retrieval_by_run[r_id] = []

        if robustness_by_run is None and self.robustness_loader:
            robustness_by_run = {}
            for r_id in run_ids:
                try:
                    robustness_by_run[r_id] = self.robustness_loader(r_id)
                except Exception:
                    robustness_by_run[r_id] = []

        return compare_runs(
            runs=runs,
            diagnostics_by_run=diagnostics_by_run,
            retrieval_by_run=retrieval_by_run,
            robustness_by_run=robustness_by_run,
        )

    def get_trend(
        self,
        experiment_id: str | None = None,
        run_ids: list[str] | None = None,
    ) -> TrendAnalysis:
        """Compute chronological metric trajectories across sequential runs."""
        if run_ids is not None:
            runs = [self.get_run(r_id) for r_id in run_ids]
        elif experiment_id is not None:
            exp_run_ids = self.repository.list_run_ids(experiment_id)
            runs = [self.get_run(r_id) for r_id in exp_run_ids]
        else:
            all_ids = self.repository.list_run_ids()
            runs = [self.get_run(r_id) for r_id in all_ids]

        return compute_trend_analysis(runs=runs, experiment_id=experiment_id)
