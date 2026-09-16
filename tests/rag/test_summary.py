"""Tests for retrieval evaluation aggregation."""

from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus
from llm_reliability.rag.summary import summarize_retrieval


def _result(
    test_case_id: str, status: RetrievalEvaluationStatus, score: float | None
) -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        evaluator_name="hit_at_3",
        test_case_id=test_case_id,
        metric_name="hit_at_k",
        status=status,
        score=score,
    )


class TestSummarizeRetrieval:
    def test_computes_mean_over_success_results_only(self):
        results = {
            "t1": [_result("t1", RetrievalEvaluationStatus.SUCCESS, 1.0)],
            "t2": [_result("t2", RetrievalEvaluationStatus.SUCCESS, 0.0)],
            "t3": [_result("t3", RetrievalEvaluationStatus.SKIPPED, None)],
        }
        summary = summarize_retrieval(results)
        stats = summary["hit_at_3"]
        assert stats.num_evaluated == 3
        assert stats.num_success == 2
        assert stats.num_skipped == 1
        assert stats.score_mean == 0.5

    def test_mrr_across_test_cases_is_the_mean_reciprocal_rank(self):
        mrr_results = {
            "t1": [
                RetrievalEvaluationResult(
                    evaluator_name="mrr",
                    test_case_id="t1",
                    metric_name="mrr",
                    status=RetrievalEvaluationStatus.SUCCESS,
                    score=1.0,
                )
            ],
            "t2": [
                RetrievalEvaluationResult(
                    evaluator_name="mrr",
                    test_case_id="t2",
                    metric_name="mrr",
                    status=RetrievalEvaluationStatus.SUCCESS,
                    score=0.5,
                )
            ],
        }
        summary = summarize_retrieval(mrr_results)
        assert summary["mrr"].score_mean == 0.75

    def test_no_success_results_leaves_score_stats_none(self):
        results = {"t1": [_result("t1", RetrievalEvaluationStatus.SKIPPED, None)]}
        stats = summarize_retrieval(results)["hit_at_3"]
        assert stats.score_mean is None
        assert stats.score_min is None
        assert stats.score_max is None

    def test_empty_input_produces_empty_summary(self):
        assert summarize_retrieval({}) == {}
