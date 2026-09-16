"""Tests for retrieval evaluators: Hit@K, Recall@K, MRR, and ground truth handling."""

import pytest

from llm_reliability.evaluation import ModelResponse, TestCase
from llm_reliability.rag.evaluators import (
    HitAtKEvaluator,
    MeanReciprocalRankEvaluator,
    RecallAtKEvaluator,
    RetrievalEvaluationStatus,
    evaluate_retrieval,
)
from llm_reliability.rag.ground_truth import get_relevant_chunk_ids


def _test_case(test_case_id: str = "t1", relevant_chunk_ids: list[str] | None = None) -> TestCase:
    metadata = {"relevant_chunk_ids": relevant_chunk_ids} if relevant_chunk_ids is not None else {}
    return TestCase(
        id=test_case_id, input="a question", reference_answer="an answer", metadata=metadata
    )


def _rag_response(test_case_id: str, retrieved_chunk_ids: list[str]) -> ModelResponse:
    return ModelResponse(
        test_case_id=test_case_id,
        output_text="an answer",
        model_id="m1",
        provider_metadata={
            "rag": {
                "query": "a question",
                "retrieved_chunks": [
                    {
                        "chunk_id": cid,
                        "document_id": "d1",
                        "text": "text",
                        "score": 1.0 - i * 0.1,
                        "rank": i + 1,
                    }
                    for i, cid in enumerate(retrieved_chunk_ids)
                ],
                "retrieval_config": {
                    "embedding_model_id": "m",
                    "top_k": len(retrieved_chunk_ids),
                    "similarity_metric": "cosine",
                },
                "pipeline_config": {},
                "prompt": "prompt text",
            }
        },
    )


def _plain_response(test_case_id: str = "t1") -> ModelResponse:
    return ModelResponse(test_case_id=test_case_id, output_text="an answer", model_id="m1")


class TestGetRelevantChunkIds:
    def test_returns_none_when_absent(self):
        assert get_relevant_chunk_ids(_test_case()) is None

    def test_returns_configured_list(self):
        tc = _test_case(relevant_chunk_ids=["c1", "c2"])
        assert get_relevant_chunk_ids(tc) == ["c1", "c2"]


class TestHitAtKEvaluator:
    def test_requires_positive_k(self):
        with pytest.raises(ValueError, match="k"):
            HitAtKEvaluator(k=0)

    def test_hit_when_relevant_chunk_in_top_k(self):
        evaluator = HitAtKEvaluator(k=2)
        tc = _test_case(relevant_chunk_ids=["c2"])
        response = _rag_response("t1", ["c1", "c2", "c3"])
        result = evaluator.evaluate(tc, response)
        assert result.status == RetrievalEvaluationStatus.SUCCESS
        assert result.score == 1.0
        assert result.metric_name == "hit_at_k"

    def test_miss_when_relevant_chunk_outside_top_k(self):
        evaluator = HitAtKEvaluator(k=2)
        tc = _test_case(relevant_chunk_ids=["c3"])
        response = _rag_response("t1", ["c1", "c2", "c3"])
        result = evaluator.evaluate(tc, response)
        assert result.score == 0.0

    def test_missing_ground_truth_is_skipped(self):
        evaluator = HitAtKEvaluator(k=2)
        result = evaluator.evaluate(_test_case(), _rag_response("t1", ["c1"]))
        assert result.status == RetrievalEvaluationStatus.SKIPPED

    def test_empty_ground_truth_list_is_skipped(self):
        evaluator = HitAtKEvaluator(k=2)
        result = evaluator.evaluate(_test_case(relevant_chunk_ids=[]), _rag_response("t1", ["c1"]))
        assert result.status == RetrievalEvaluationStatus.SKIPPED

    def test_non_rag_response_is_skipped(self):
        evaluator = HitAtKEvaluator(k=2)
        tc = _test_case(relevant_chunk_ids=["c1"])
        result = evaluator.evaluate(tc, _plain_response())
        assert result.status == RetrievalEvaluationStatus.SKIPPED

    def test_name_reflects_k(self):
        assert HitAtKEvaluator(k=5).name == "hit_at_5"


class TestRecallAtKEvaluator:
    def test_recall_with_single_relevant_target_matches_hit_at_k(self):
        tc = _test_case(relevant_chunk_ids=["c2"])
        response = _rag_response("t1", ["c1", "c2", "c3"])
        hit = HitAtKEvaluator(k=2).evaluate(tc, response)
        recall = RecallAtKEvaluator(k=2).evaluate(tc, response)
        assert hit.score == recall.score

    def test_recall_with_multiple_relevant_targets_is_fractional(self):
        tc = _test_case(relevant_chunk_ids=["c1", "c2", "c3"])
        response = _rag_response("t1", ["c1", "c4", "c2"])
        result = RecallAtKEvaluator(k=3).evaluate(tc, response)
        assert result.score == pytest.approx(2 / 3)

    def test_recall_all_targets_found_is_one(self):
        tc = _test_case(relevant_chunk_ids=["c1", "c2"])
        response = _rag_response("t1", ["c1", "c2"])
        result = RecallAtKEvaluator(k=2).evaluate(tc, response)
        assert result.score == 1.0

    def test_recall_none_found_is_zero(self):
        tc = _test_case(relevant_chunk_ids=["c1", "c2"])
        response = _rag_response("t1", ["c3", "c4"])
        result = RecallAtKEvaluator(k=2).evaluate(tc, response)
        assert result.score == 0.0

    def test_missing_ground_truth_is_skipped(self):
        result = RecallAtKEvaluator(k=2).evaluate(_test_case(), _rag_response("t1", ["c1"]))
        assert result.status == RetrievalEvaluationStatus.SKIPPED


class TestMeanReciprocalRankEvaluator:
    def test_reciprocal_of_first_relevant_rank(self):
        tc = _test_case(relevant_chunk_ids=["c3"])
        response = _rag_response("t1", ["c1", "c2", "c3", "c4"])
        result = MeanReciprocalRankEvaluator().evaluate(tc, response)
        assert result.score == pytest.approx(1 / 3)

    def test_first_rank_relevant_scores_one(self):
        tc = _test_case(relevant_chunk_ids=["c1"])
        response = _rag_response("t1", ["c1", "c2"])
        result = MeanReciprocalRankEvaluator().evaluate(tc, response)
        assert result.score == 1.0

    def test_no_relevant_chunk_retrieved_scores_zero(self):
        tc = _test_case(relevant_chunk_ids=["c9"])
        response = _rag_response("t1", ["c1", "c2"])
        result = MeanReciprocalRankEvaluator().evaluate(tc, response)
        assert result.score == 0.0

    def test_missing_ground_truth_is_skipped(self):
        result = MeanReciprocalRankEvaluator().evaluate(_test_case(), _rag_response("t1", ["c1"]))
        assert result.status == RetrievalEvaluationStatus.SKIPPED


class TestEvaluateRetrieval:
    def test_runs_all_evaluators_for_all_test_cases_with_a_response(self):
        test_cases = [
            _test_case("t1", relevant_chunk_ids=["c1"]),
            _test_case("t2", relevant_chunk_ids=["c9"]),
        ]
        responses = {
            "t1": _rag_response("t1", ["c1", "c2"]),
            "t2": _rag_response("t2", ["c1", "c2"]),
        }
        results = evaluate_retrieval(
            test_cases, responses, [HitAtKEvaluator(k=2), MeanReciprocalRankEvaluator()]
        )
        assert set(results.keys()) == {"t1", "t2"}
        assert len(results["t1"]) == 2

    def test_test_case_without_response_is_skipped_entirely(self):
        test_cases = [_test_case("t1", relevant_chunk_ids=["c1"])]
        results = evaluate_retrieval(test_cases, {}, [HitAtKEvaluator(k=2)])
        assert results == {}

    def test_evaluator_exception_is_isolated_as_execution_error(self):
        class RaisingEvaluator(HitAtKEvaluator):
            def evaluate(self, test_case, response):
                raise RuntimeError("boom")

        test_cases = [_test_case("t1", relevant_chunk_ids=["c1"])]
        responses = {"t1": _rag_response("t1", ["c1"])}
        results = evaluate_retrieval(test_cases, responses, [RaisingEvaluator(k=1)])
        assert results["t1"][0].status == RetrievalEvaluationStatus.EXECUTION_ERROR
        assert "boom" in results["t1"][0].error
