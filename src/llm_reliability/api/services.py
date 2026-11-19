"""
Application services bridging domain models, diagnostics, and API contracts.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.api.schemas.common import redact_sensitive_dict
from llm_reliability.api.schemas.diagnostics import (
    ClaimAssessmentResponse,
    ReliabilityDiagnosticResponse,
)
from llm_reliability.api.schemas.experiments import (
    DatasetVersionResponse,
    EvaluatorConfigResponse,
    ExperimentDetailResponse,
    ExperimentSummaryResponse,
    ModelConfigResponse,
)
from llm_reliability.api.schemas.rag import (
    RagTestCaseResultResponse,
    RetrievalEvaluationResultResponse,
    RetrievedChunkResponse,
)
from llm_reliability.api.schemas.robustness import (
    RobustnessBundleResponse,
    RobustnessComparisonResponse,
    RobustnessConsistencyResponse,
)
from llm_reliability.api.schemas.runs import (
    EvaluationResultResponse,
    RunDetailResponse,
    RunSummariesResponse,
    RunSummaryResponse,
    TestCaseResultResponse,
)
from llm_reliability.diagnostics.diagnosis import (
    ReliabilityDiagnostic,
    build_reliability_diagnostic,
)
from llm_reliability.diagnostics.evidence import assess_statements_lexically
from llm_reliability.diagnostics.robustness import (
    RobustnessResult,
    compare_robustness,
    get_perturbation_info,
    summarize_robustness,
)
from llm_reliability.experiments.models import ExperimentDefinition, ExperimentRun
from llm_reliability.rag.evaluators import (
    HitAtKEvaluator,
    MeanReciprocalRankEvaluator,
    RecallAtKEvaluator,
    RetrievalEvaluationResult,
    evaluate_retrieval,
)
from llm_reliability.rag.summary import summarize_retrieval
from llm_reliability.reliability.findings import analyze_reliability
from llm_reliability.reliability.summary import summarize_run


class DiagnosticsService:
    """Derives forensic reliability diagnostics, retrieval metrics, and robustness comparisons."""

    def __init__(self) -> None:
        self._ret_evaluators = [
            HitAtKEvaluator(k=1),
            RecallAtKEvaluator(k=1),
            MeanReciprocalRankEvaluator(),
        ]
        self._diag_cache: dict[str, dict[str, ReliabilityDiagnostic]] = {}
        self._ret_cache: dict[str, dict[str, list[RetrievalEvaluationResult]]] = {}
        self._rob_cache: dict[str, list[RobustnessResult]] = {}

    def get_retrieval_results(
        self, run: ExperimentRun
    ) -> dict[str, list[RetrievalEvaluationResult]]:
        if run.run_id in self._ret_cache:
            return self._ret_cache[run.run_id]

        if not run.result:
            return {}

        has_rag = any(
            tc.model_response and "rag" in tc.model_response.provider_metadata
            for tc in run.result.test_case_results
        )
        if not has_rag:
            self._ret_cache[run.run_id] = {}
            return {}

        responses = {
            tc.test_case_id: tc.model_response
            for tc in run.result.test_case_results
            if tc.model_response is not None
        }
        res = evaluate_retrieval(
            run.dataset_version.dataset.test_cases,
            responses,
            self._ret_evaluators,
        )
        self._ret_cache[run.run_id] = res
        return res

    def get_diagnostics(self, run: ExperimentRun) -> dict[str, ReliabilityDiagnostic]:
        if run.run_id in self._diag_cache:
            return self._diag_cache[run.run_id]

        if not run.result:
            return {}

        ret_results = self.get_retrieval_results(run)
        tc_map = {tc.id: tc for tc in run.dataset_version.dataset.test_cases}
        res_map = {tc.test_case_id: tc for tc in run.result.test_case_results}

        diags: dict[str, ReliabilityDiagnostic] = {}
        for tc_id, test_case in tc_map.items():
            tc_res = res_map.get(tc_id)
            if tc_res is None:
                continue
            case_ret = ret_results.get(tc_id, [])
            diag = build_reliability_diagnostic(test_case, tc_res, case_ret)
            if diag.retrieved_context and diag.generated_answer:
                diag.claim_assessments = assess_statements_lexically(
                    diag.generated_answer, diag.retrieved_context
                )
            diags[tc_id] = diag

        self._diag_cache[run.run_id] = diags
        return diags

    def get_robustness(self, run: ExperimentRun) -> list[RobustnessResult]:
        if run.run_id in self._rob_cache:
            return self._rob_cache[run.run_id]

        if not run.result:
            return []

        diags = self.get_diagnostics(run)
        tc_map = {tc.id: tc for tc in run.dataset_version.dataset.test_cases}
        rob_results: list[RobustnessResult] = []

        for tc_id, test_case in tc_map.items():
            info = get_perturbation_info(test_case.metadata)
            if info is None:
                continue
            baseline_id, pert_type = info
            if baseline_id in diags and tc_id in diags:
                rob_results.append(compare_robustness(diags[baseline_id], diags[tc_id], pert_type))

        self._rob_cache[run.run_id] = rob_results
        return rob_results

    def get_run_summaries(self, run: ExperimentRun) -> RunSummariesResponse:
        eval_sum = summarize_run(run.result).to_dict() if run.result else {}
        rel_findings = [f.to_dict() for f in analyze_reliability(run.result)] if run.result else []
        ret_results = self.get_retrieval_results(run)
        ret_sum = (
            {name: s.to_dict() for name, s in summarize_retrieval(ret_results).items()}
            if ret_results
            else {}
        )
        return RunSummariesResponse(
            run_id=run.run_id,
            evaluation_summary=eval_sum,
            reliability_findings=rel_findings,
            retrieval_summary=ret_sum,
        )


def to_experiment_summary(defn: ExperimentDefinition, runs_count: int) -> ExperimentSummaryResponse:
    return ExperimentSummaryResponse(
        experiment_id=defn.experiment_id,
        name=defn.name,
        description=defn.description,
        created_at=defn.created_at,
        config_fingerprint=defn.config_fingerprint,
        dataset_name=defn.dataset_version.dataset_name,
        test_case_count=len(defn.dataset_version.dataset.test_cases),
        model_id=defn.model_config.model_id,
        evaluator_count=len(defn.evaluator_configs),
        runs_count=runs_count,
    )


def to_experiment_detail(defn: ExperimentDefinition, runs_count: int) -> ExperimentDetailResponse:
    return ExperimentDetailResponse(
        experiment_id=defn.experiment_id,
        name=defn.name,
        description=defn.description,
        created_at=defn.created_at,
        config_fingerprint=defn.config_fingerprint,
        dataset_version=DatasetVersionResponse(
            version_id=defn.dataset_version.version_id,
            dataset_name=defn.dataset_version.dataset_name,
            test_case_count=len(defn.dataset_version.dataset.test_cases),
        ),
        model_config=ModelConfigResponse(
            model_id=defn.model_config.model_id,
            temperature=defn.model_config.temperature,
            max_tokens=defn.model_config.max_tokens,
            extra_params=redact_sensitive_dict(defn.model_config.extra_params),
            metadata=redact_sensitive_dict(defn.model_config.extra_params),
        ),
        evaluator_configs=[
            EvaluatorConfigResponse(
                evaluator_name=ec.evaluator_name,
                criterion=str(ec.parameters.get("criterion", ec.evaluator_name)),
                evaluator_version=str(ec.parameters.get("evaluator_version", "1.0.0")),
                config=redact_sensitive_dict(ec.parameters),
                parameters=redact_sensitive_dict(ec.parameters),
            )
            for ec in defn.evaluator_configs
        ],
        metadata=redact_sensitive_dict(defn.metadata),
        runs_count=runs_count,
    )


def to_run_summary(run: ExperimentRun) -> RunSummaryResponse:
    dur = None
    if run.started_at and run.finished_at:
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(run.started_at)
            t1 = datetime.fromisoformat(run.finished_at)
            dur = max(0.0, (t1 - t0).total_seconds())
        except Exception:
            pass

    return RunSummaryResponse(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_seconds=dur,
        config_fingerprint=run.config_fingerprint,
        num_test_cases=len(run.result.test_case_results) if run.result else 0,
        num_succeeded=run.result.num_succeeded if run.result else 0,
        num_failed=run.result.num_failed if run.result else 0,
        num_passed=run.result.num_passed if run.result else 0,
        pass_rate=run.result.pass_rate if run.result else 0.0,
        error=run.error,
    )


def to_run_detail(run: ExperimentRun, include_test_cases: bool = True) -> RunDetailResponse:
    dur = None
    if run.started_at and run.finished_at:
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(run.started_at)
            t1 = datetime.fromisoformat(run.finished_at)
            dur = max(0.0, (t1 - t0).total_seconds())
        except Exception:
            pass

    test_case_results: list[TestCaseResultResponse] = []
    if include_test_cases and run.result:
        for tc in run.result.test_case_results:
            evals = [
                EvaluationResultResponse(
                    evaluator_name=e.evaluator_name,
                    criterion=e.criterion.value
                    if (e.criterion is not None and hasattr(e.criterion, "value"))
                    else str(e.criterion or ""),
                    passed=e.passed,
                    score=e.score,
                    label=e.label,
                    explanation=e.explanation,
                    status=e.status.value if hasattr(e.status, "value") else str(e.status),
                    error=e.error,
                    duration_seconds=e.duration_seconds,
                )
                for e in tc.evaluation_results
            ]
            test_case_results.append(
                TestCaseResultResponse(
                    test_case_id=tc.test_case_id,
                    input_text=tc.input_text,
                    reference_answer=tc.reference_answer,
                    output_text=tc.model_response.output_text if tc.model_response else None,
                    passed=tc.passed,
                    succeeded=tc.succeeded,
                    evaluations=evals,
                    evaluation_results=evals,
                    duration_seconds=tc.duration_seconds,
                    execution_error=tc.execution_error,
                    provider_metadata=redact_sensitive_dict(
                        tc.model_response.provider_metadata if tc.model_response else {}
                    ),
                )
            )

    return RunDetailResponse(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_seconds=dur,
        config_fingerprint=run.config_fingerprint,
        dataset_version=DatasetVersionResponse(
            version_id=run.dataset_version.version_id,
            dataset_name=run.dataset_version.dataset_name,
            test_case_count=len(run.dataset_version.dataset.test_cases),
        ),
        model_config=ModelConfigResponse(
            model_id=run.model_config.model_id,
            temperature=run.model_config.temperature,
            max_tokens=run.model_config.max_tokens,
            extra_params=redact_sensitive_dict(run.model_config.extra_params),
            metadata=redact_sensitive_dict(run.model_config.extra_params),
        ),
        evaluator_configs=[
            EvaluatorConfigResponse(
                evaluator_name=ec.evaluator_name,
                criterion=str(ec.parameters.get("criterion", ec.evaluator_name)),
                evaluator_version=str(ec.parameters.get("evaluator_version", "1.0.0")),
                config=redact_sensitive_dict(ec.parameters),
                parameters=redact_sensitive_dict(ec.parameters),
            )
            for ec in run.evaluator_configs
        ],
        num_test_cases=len(run.result.test_case_results) if run.result else 0,
        num_succeeded=run.result.num_succeeded if run.result else 0,
        num_failed=run.result.num_failed if run.result else 0,
        num_passed=run.result.num_passed if run.result else 0,
        pass_rate=run.result.pass_rate if run.result else 0.0,
        test_case_results=test_case_results,
        error=run.error,
    )


def to_diagnostic_response(
    diag: ReliabilityDiagnostic, run_id: str
) -> ReliabilityDiagnosticResponse:
    claims = [
        ClaimAssessmentResponse(
            statement=c.statement,
            status=c.assessment.status.value
            if hasattr(c.assessment.status, "value")
            else str(c.assessment.status),
            method=c.assessment.method,
            confidence=c.assessment.confidence,
            evidence_snippets=list(c.assessment.supporting_evidence),
            reasoning=c.assessment.explanation,
        )
        for c in diag.claim_assessments
    ]
    return ReliabilityDiagnosticResponse(
        test_case_id=diag.test_case_id,
        run_id=run_id,
        failure_category=diag.failure_category.value
        if (diag.failure_category is not None and hasattr(diag.failure_category, "value"))
        else (str(diag.failure_category) if diag.failure_category is not None else None),
        secondary_findings=list(diag.secondary_findings),
        status=diag.status.value if hasattr(diag.status, "value") else str(diag.status),
        confidence=diag.confidence,
        explanation=diag.explanation,
        question=diag.question,
        reference_answer=diag.reference_answer,
        generated_answer=diag.generated_answer,
        retrieved_context=diag.retrieved_context,
        contributing_evaluators=list(diag.contributing_evaluators),
        supporting_evidence=list(diag.supporting_evidence),
        claim_assessments=claims,
        error=diag.error,
    )


def to_rag_response(
    test_case_id: str,
    run_id: str,
    tc_res: Any,
    ret_evals: list[RetrievalEvaluationResult],
) -> RagTestCaseResultResponse:
    chunks: list[RetrievedChunkResponse] = []
    if tc_res.model_response and "rag" in tc_res.model_response.provider_metadata:
        rag_meta = tc_res.model_response.provider_metadata["rag"]
        raw_chunks = rag_meta.get("retrieved_chunks", [])
        for rc in raw_chunks:
            chunks.append(
                RetrievedChunkResponse(
                    chunk_id=rc.get("chunk_id") or rc.get("id", ""),
                    document_id=rc.get("document_id", ""),
                    text=rc.get("text") or rc.get("content", ""),
                    similarity_score=rc.get("similarity_score", rc.get("score")),
                    rank=rc.get("rank", 0),
                )
            )

    eval_responses = [
        RetrievalEvaluationResultResponse(
            evaluator_name=re.evaluator_name,
            metric_name=re.metric_name,
            score=re.score,
            k=re.evaluator_config.get("k") if isinstance(re.evaluator_config, dict) else None,
            status=re.status.value if hasattr(re.status, "value") else str(re.status),
            ground_truth_chunk_ids=list(re.relevant_targets)
            if re.relevant_targets is not None
            else [],
            retrieved_chunk_ids=list(re.retrieved_ids) if re.retrieved_ids is not None else [],
            evaluator_config=redact_sensitive_dict(re.evaluator_config),
            error=re.error,
        )
        for re in ret_evals
    ]

    return RagTestCaseResultResponse(
        test_case_id=test_case_id,
        run_id=run_id,
        question=tc_res.input_text,
        generated_answer=tc_res.model_response.output_text if tc_res.model_response else None,
        retrieved_chunks=chunks,
        retrieval_evaluations=eval_responses,
    )


def to_robustness_bundle(
    run_id: str, comparisons: list[RobustnessResult]
) -> RobustnessBundleResponse:
    def _val(x: Any) -> str | None:
        if x is None:
            return None
        return x.value if hasattr(x, "value") else str(x)

    items = [
        RobustnessComparisonResponse(
            baseline_test_case_id=r.baseline_test_case_id,
            perturbed_test_case_id=r.perturbed_test_case_id,
            perturbation_type=_val(r.perturbation_type) or "",
            baseline_category=_val(r.baseline_category),
            perturbed_category=_val(r.perturbed_category),
            outcome_changed=r.outcome_changed,
            category_changed=(r.baseline_category != r.perturbed_category),
            failure_category_transition=r.failure_category_transition,
        )
        for r in comparisons
    ]
    consistency_resp = None
    if items:
        consistency = summarize_robustness(comparisons)
        consistency_resp = RobustnessConsistencyResponse(
            total_comparisons=consistency.num_comparisons,
            identical_outcomes=consistency.num_unchanged,
            consistency_rate=consistency.outcome_consistency_rate,
            category_transitions=dict(consistency.transition_counts),
            transitions_by_type={},
        )
    return RobustnessBundleResponse(
        run_id=run_id,
        comparisons=items,
        consistency=consistency_resp,
    )
