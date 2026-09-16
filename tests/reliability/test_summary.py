"""Tests for evaluation summary aggregation."""

from llm_reliability.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    RunResult,
    TestCaseResult,
)
from llm_reliability.reliability import summarize_run


def _tc_result(test_case_id: str, evaluation_results: list[EvaluationResult]) -> TestCaseResult:
    return TestCaseResult(
        test_case_id=test_case_id,
        input_text="q",
        reference_answer="a",
        model_response=ModelResponse(test_case_id=test_case_id, output_text="a", model_id="m"),
        evaluation_results=evaluation_results,
    )


def _exact_match(test_case_id: str, passed: bool) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="exact_match",
        test_case_id=test_case_id,
        status=EvaluationStatus.SUCCESS,
        criterion="correctness",
        score=1.0 if passed else 0.0,
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


class TestSummarizeRun:
    def test_counts_success_skipped_and_error_per_evaluator(self):
        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["exact_match"],
            test_case_results=[
                _tc_result("t1", [_exact_match("t1", True)]),
                _tc_result("t2", [_skipped("exact_match", "t2")]),
                _tc_result("t3", [_error("exact_match", "t3")]),
            ],
        )
        summary = summarize_run(run_result)
        stats = summary.per_evaluator["exact_match"]
        assert stats.num_evaluated == 3
        assert stats.num_success == 1
        assert stats.num_skipped == 1
        assert stats.num_error == 1

    def test_score_statistics_ignore_none_scores(self):
        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["exact_match"],
            test_case_results=[
                _tc_result("t1", [_exact_match("t1", True)]),
                _tc_result("t2", [_exact_match("t2", False)]),
                _tc_result("t3", [_skipped("exact_match", "t3")]),
            ],
        )
        stats = summarize_run(run_result).per_evaluator["exact_match"]
        assert stats.score_count == 2
        assert stats.score_mean == 0.5
        assert stats.score_min == 0.0
        assert stats.score_max == 1.0

    def test_purely_categorical_evaluator_has_no_score_statistics(self):
        judge_result = EvaluationResult(
            evaluator_name="llm_judge",
            test_case_id="t1",
            status=EvaluationStatus.SUCCESS,
            criterion="correctness",
            passed=True,
            label="correct",
        )
        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["llm_judge"],
            test_case_results=[_tc_result("t1", [judge_result])],
        )
        stats = summarize_run(run_result).per_evaluator["llm_judge"]
        assert stats.score_count == 0
        assert stats.score_mean is None
        assert stats.score_min is None
        assert stats.score_max is None

    def test_disagreement_count_detects_split_passed_values(self):
        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["exact_match", "semantic_similarity"],
            test_case_results=[
                _tc_result(
                    "t1",
                    [
                        _exact_match("t1", False),
                        EvaluationResult(
                            evaluator_name="semantic_similarity",
                            test_case_id="t1",
                            status=EvaluationStatus.SUCCESS,
                            criterion="correctness",
                            score=0.9,
                            passed=True,
                        ),
                    ],
                ),
                _tc_result(
                    "t2",
                    [
                        _exact_match("t2", True),
                        EvaluationResult(
                            evaluator_name="semantic_similarity",
                            test_case_id="t2",
                            status=EvaluationStatus.SUCCESS,
                            criterion="correctness",
                            score=0.95,
                            passed=True,
                        ),
                    ],
                ),
            ],
        )
        summary = summarize_run(run_result)
        assert summary.disagreement_count == 1

    def test_summary_never_produces_a_single_overall_score(self):
        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["exact_match"],
            test_case_results=[_tc_result("t1", [_exact_match("t1", True)])],
        )
        summary_dict = summarize_run(run_result).to_dict()
        assert "overall_score" not in summary_dict
        assert "score" not in summary_dict

    def test_to_dict_is_json_serializable(self):
        import json

        run_result = RunResult(
            dataset_name="ds",
            model_id="m",
            evaluator_names=["exact_match"],
            test_case_results=[_tc_result("t1", [_exact_match("t1", True)])],
        )
        json.dumps(summarize_run(run_result).to_dict())
