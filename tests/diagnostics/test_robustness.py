"""Tests for baseline-versus-perturbed robustness comparison and aggregation."""

from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
from llm_reliability.diagnostics.robustness import (
    BASELINE_TEST_CASE_ID_KEY,
    PERTURBATION_TYPE_KEY,
    PerturbationType,
    compare_robustness,
    get_perturbation_info,
    summarize_robustness,
)
from llm_reliability.diagnostics.taxonomy import DiagnosticStatus, FailureCategory


def _diagnostic(
    test_case_id: str, category: FailureCategory | None, status=DiagnosticStatus.DETERMINED
) -> ReliabilityDiagnostic:
    return ReliabilityDiagnostic(
        test_case_id=test_case_id,
        question="q",
        generated_answer="a",
        reference_answer="a",
        retrieved_context=None,
        status=status,
        failure_category=category,
    )


class TestGetPerturbationInfo:
    def test_returns_none_when_not_a_perturbation(self):
        assert get_perturbation_info({}) is None

    def test_returns_baseline_id_and_type(self):
        metadata = {BASELINE_TEST_CASE_ID_KEY: "t1", PERTURBATION_TYPE_KEY: "question_rewording"}
        assert get_perturbation_info(metadata) == ("t1", PerturbationType.QUESTION_REWORDING)


class TestCompareRobustness:
    def test_unchanged_outcome(self):
        baseline = _diagnostic("t1", FailureCategory.CORRECT)
        perturbed = _diagnostic("t1-reworded", FailureCategory.CORRECT)
        result = compare_robustness(baseline, perturbed, PerturbationType.QUESTION_REWORDING)
        assert result.outcome_changed is False
        assert result.failure_category_transition is None

    def test_changed_outcome_records_transition(self):
        baseline = _diagnostic("t1", FailureCategory.CORRECT)
        perturbed = _diagnostic("t1-reworded", FailureCategory.UNSUPPORTED)
        result = compare_robustness(baseline, perturbed, PerturbationType.QUESTION_REWORDING)
        assert result.outcome_changed is True
        assert result.failure_category_transition == "correct -> unsupported"

    def test_none_category_counts_as_a_distinct_outcome(self):
        baseline = _diagnostic("t1", FailureCategory.CORRECT)
        perturbed = _diagnostic("t1-reworded", None)
        result = compare_robustness(baseline, perturbed, PerturbationType.TOP_K_CHANGE)
        assert result.outcome_changed is True

    def test_round_trip_through_dict(self):
        from llm_reliability.diagnostics.robustness import RobustnessResult

        baseline = _diagnostic("t1", FailureCategory.CORRECT)
        perturbed = _diagnostic("t1-r", FailureCategory.UNSUPPORTED)
        result = compare_robustness(baseline, perturbed, PerturbationType.CONTEXT_REORDERING)
        reloaded = RobustnessResult.from_dict(result.to_dict())
        assert reloaded == result


class TestSummarizeRobustness:
    def test_consistency_rate_computed_correctly(self):
        results = [
            compare_robustness(
                _diagnostic("a", FailureCategory.CORRECT),
                _diagnostic("a-r", FailureCategory.CORRECT),
                PerturbationType.QUESTION_REWORDING,
            ),
            compare_robustness(
                _diagnostic("b", FailureCategory.CORRECT),
                _diagnostic("b-r", FailureCategory.UNSUPPORTED),
                PerturbationType.QUESTION_REWORDING,
            ),
        ]
        summary = summarize_robustness(results)
        assert summary.num_comparisons == 2
        assert summary.num_unchanged == 1
        assert summary.outcome_consistency_rate == 0.5

    def test_empty_input_has_no_rate_not_a_fabricated_zero(self):
        summary = summarize_robustness([])
        assert summary.num_comparisons == 0
        assert summary.outcome_consistency_rate is None

    def test_transition_counts_are_tracked(self):
        results = [
            compare_robustness(
                _diagnostic("a", FailureCategory.CORRECT),
                _diagnostic("a-r", FailureCategory.UNSUPPORTED),
                PerturbationType.QUESTION_REWORDING,
            ),
            compare_robustness(
                _diagnostic("b", FailureCategory.CORRECT),
                _diagnostic("b-r", FailureCategory.UNSUPPORTED),
                PerturbationType.QUESTION_REWORDING,
            ),
        ]
        summary = summarize_robustness(results)
        assert summary.transition_counts == {"correct -> unsupported": 2}
