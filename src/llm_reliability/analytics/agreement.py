"""
Pairwise evaluator agreement and disagreement analytics.

Quantifies concordance between compatible evaluators assessing the same
criterion, preserving the exact cases responsible for disagreement.
"""

from __future__ import annotations

import itertools

from llm_reliability.analytics.models import EvaluatorAgreementSummary
from llm_reliability.evaluation.models import EvaluationResult, EvaluationStatus
from llm_reliability.experiments.models import ExperimentRun


def compute_evaluator_agreement(
    run: ExperimentRun, evaluator_a: str, evaluator_b: str
) -> EvaluatorAgreementSummary | None:
    """Calculate agreement statistics between two evaluators on compatible criteria.

    Returns ``None`` if either evaluator is missing from the run, or if the
    evaluators assess incompatible criteria (e.g. comparing answer correctness
    directly against context support without an explicit mapping).
    """
    if run.result is None:
        return None

    test_cases = run.result.test_case_results

    criterion_a: str | None = None
    criterion_b: str | None = None

    pairs: list[tuple[str, EvaluationResult, EvaluationResult]] = []

    for tc in test_cases:
        res_a = next((r for r in tc.evaluation_results if r.evaluator_name == evaluator_a), None)
        res_b = next((r for r in tc.evaluation_results if r.evaluator_name == evaluator_b), None)

        if res_a is None or res_b is None:
            continue

        if criterion_a is None and res_a.criterion:
            criterion_a = res_a.criterion
        if criterion_b is None and res_b.criterion:
            criterion_b = res_b.criterion

        pairs.append((tc.test_case_id, res_a, res_b))

    if not pairs:
        return None

    # Criteria compatibility check: do not treat distinct criteria as equivalent
    if criterion_a is not None and criterion_b is not None and criterion_a != criterion_b:
        return None

    criterion = criterion_a or criterion_b or "unspecified"

    cases_compared = 0
    agreements = 0
    disagreements = 0
    excluded_cases = 0
    disagreement_ids: list[str] = []

    for tc_id, r_a, r_b in pairs:
        if (
            r_a.status != EvaluationStatus.SUCCESS
            or r_b.status != EvaluationStatus.SUCCESS
            or r_a.passed is None
            or r_b.passed is None
        ):
            excluded_cases += 1
            continue

        cases_compared += 1
        if r_a.passed == r_b.passed:
            agreements += 1
        else:
            disagreements += 1
            disagreement_ids.append(tc_id)

    agreement_rate = (agreements / cases_compared) if cases_compared > 0 else None

    return EvaluatorAgreementSummary(
        evaluator_a=evaluator_a,
        evaluator_b=evaluator_b,
        criterion=criterion,
        cases_compared=cases_compared,
        agreements=agreements,
        disagreements=disagreements,
        agreement_rate=agreement_rate,
        excluded_cases=excluded_cases,
        disagreement_test_case_ids=disagreement_ids,
    )


def compute_all_pairwise_agreements(run: ExperimentRun) -> list[EvaluatorAgreementSummary]:
    """Compute pairwise agreement summaries across all compatible evaluator pairs in a run."""
    if run.result is None:
        return []

    evaluator_names = sorted(run.result.evaluator_names)
    summaries: list[EvaluatorAgreementSummary] = []

    for ev_a, ev_b in itertools.combinations(evaluator_names, 2):
        summary = compute_evaluator_agreement(run, ev_a, ev_b)
        if summary is not None:
            summaries.append(summary)

    return summaries
