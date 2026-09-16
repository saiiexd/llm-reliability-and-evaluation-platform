"""Tests for reliability findings derived from evaluation evidence."""

from llm_reliability.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    RunResult,
    TestCaseResult,
)
from llm_reliability.reliability import ReliabilityFlagType, analyze_reliability


def _tc_result(
    test_case_id: str,
    evaluation_results: list[EvaluationResult],
    execution_error: str | None = None,
) -> TestCaseResult:
    return TestCaseResult(
        test_case_id=test_case_id,
        input_text="q",
        reference_answer="a",
        model_response=(
            None
            if execution_error
            else ModelResponse(test_case_id=test_case_id, output_text="a", model_id="m")
        ),
        evaluation_results=evaluation_results,
        execution_error=execution_error,
    )


def _run(test_case_results: list[TestCaseResult]) -> RunResult:
    return RunResult(
        dataset_name="ds",
        model_id="m",
        evaluator_names=["exact_match"],
        test_case_results=test_case_results,
    )


def _success(evaluator_name: str, test_case_id: str, passed: bool) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name=evaluator_name,
        test_case_id=test_case_id,
        status=EvaluationStatus.SUCCESS,
        criterion="correctness",
        passed=passed,
    )


def _skipped(evaluator_name: str, test_case_id: str) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name=evaluator_name,
        test_case_id=test_case_id,
        status=EvaluationStatus.SKIPPED,
        criterion="correctness",
        label="no_reference",
    )


def _error(evaluator_name: str, test_case_id: str) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name=evaluator_name,
        test_case_id=test_case_id,
        status=EvaluationStatus.EXECUTION_ERROR,
        criterion="correctness",
        error="boom",
    )


class TestAnalyzeReliability:
    def test_no_findings_for_clean_agreeing_success(self):
        run_result = _run([_tc_result("t1", [_success("exact_match", "t1", True)])])
        findings = analyze_reliability(run_result)
        assert findings == []

    def test_disagreement_is_flagged(self):
        run_result = _run(
            [
                _tc_result(
                    "t1",
                    [
                        _success("exact_match", "t1", False),
                        _success("semantic_similarity", "t1", True),
                    ],
                )
            ]
        )
        findings = analyze_reliability(run_result)
        flag_types = {f.flag_type for f in findings}
        assert ReliabilityFlagType.EVALUATOR_DISAGREEMENT in flag_types

    def test_missing_evaluation_evidence_when_all_evaluators_skipped(self):
        run_result = _run(
            [
                _tc_result(
                    "t1", [_skipped("exact_match", "t1"), _skipped("semantic_similarity", "t1")]
                )
            ]
        )
        findings = analyze_reliability(run_result)
        assert len(findings) == 1
        assert findings[0].flag_type == ReliabilityFlagType.MISSING_EVALUATION_EVIDENCE

    def test_partial_skip_is_not_missing_evidence(self):
        run_result = _run(
            [
                _tc_result(
                    "t1",
                    [_skipped("exact_match", "t1"), _success("semantic_similarity", "t1", True)],
                )
            ]
        )
        findings = analyze_reliability(run_result)
        flag_types = {f.flag_type for f in findings}
        assert ReliabilityFlagType.MISSING_EVALUATION_EVIDENCE not in flag_types

    def test_evaluator_failure_is_flagged(self):
        run_result = _run(
            [
                _tc_result(
                    "t1", [_error("exact_match", "t1"), _success("semantic_similarity", "t1", True)]
                )
            ]
        )
        findings = analyze_reliability(run_result)
        failure_findings = [
            f for f in findings if f.flag_type == ReliabilityFlagType.EVALUATOR_FAILURE
        ]
        assert len(failure_findings) == 1
        assert failure_findings[0].evaluator_names == ("exact_match",)

    def test_incorrect_per_evaluator_is_flagged_but_not_labeled_hallucination(self):
        run_result = _run([_tc_result("t1", [_success("exact_match", "t1", False)])])
        findings = analyze_reliability(run_result)
        incorrect = [
            f for f in findings if f.flag_type == ReliabilityFlagType.INCORRECT_PER_EVALUATOR
        ]
        assert len(incorrect) == 1
        assert (
            "hallucination" not in incorrect[0].detail.lower()
            or "not a hallucination" in incorrect[0].detail.lower()
        )

    def test_failed_execution_produces_no_evaluator_findings(self):
        run_result = _run([_tc_result("t1", [], execution_error="AdapterError: boom")])
        findings = analyze_reliability(run_result)
        assert findings == []

    def test_findings_are_json_serializable(self):
        import json

        run_result = _run([_tc_result("t1", [_success("exact_match", "t1", False)])])
        findings = analyze_reliability(run_result)
        json.dumps([f.to_dict() for f in findings])

    def test_no_finding_type_is_named_hallucination(self):
        for flag_type in ReliabilityFlagType:
            assert "hallucination" not in flag_type.value
