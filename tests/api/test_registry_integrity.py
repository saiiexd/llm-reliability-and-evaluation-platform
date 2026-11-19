"""
Regression tests for the live API registries, exercised end to end.

The unit tests for each evaluator used hand-written fakes, so they never
noticed that the fake judge wired into the *API* registry (and the seed
script) replied with keys the strict judge parser rejects. Every faithfulness
result in the live app was therefore an ``output_validation_error`` and every
RAG case was counted as failed. These tests go through the real registries.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_reliability.api.registry import JUDGE_RESPONSES
from llm_reliability.evaluation.llm_judge import parse_judge_output
from llm_reliability.rag.faithfulness_judge import FAITHFULNESS_RUBRIC

API = "/api/v1"


def test_canned_judge_replies_conform_to_faithfulness_rubric() -> None:
    for test_case_id, raw in JUDGE_RESPONSES.items():
        parsed = parse_judge_output(raw, FAITHFULNESS_RUBRIC)
        assert parsed.outcome in FAITHFULNESS_RUBRIC.scale, test_case_id


def test_seeded_runs_have_no_judge_validation_errors(api_client: TestClient) -> None:
    run = api_client.get(f"{API}/runs/run-rag-baseline-001").json()
    judge_results = [
        e
        for tc in run["test_case_results"]
        for e in tc["evaluations"]
        if e["evaluator_name"] == "llm_faithfulness_judge"
    ]
    assert judge_results, "seeded RAG run should include faithfulness judge results"
    assert all(e["status"] == "success" for e in judge_results), judge_results
    # The correct, supported answer must now be able to pass overall.
    by_id = {tc["test_case_id"]: tc for tc in run["test_case_results"]}
    assert by_id["correct-context-correct-answer"]["passed"] is True
    assert run["num_passed"] > 0


def test_retrieval_metric_rejected_as_answer_evaluator(api_client: TestClient) -> None:
    response = api_client.post(
        f"{API}/experiments",
        json={
            "experiment_id": "exp-retrieval-as-answer-eval",
            "name": "x",
            "dataset_name": "d",
            "test_cases": [{"id": "tc-1", "input": "q", "reference_answer": "a"}],
            "model_config": {"model_id": "rag-pipeline-v1"},
            "evaluator_names": ["exact_match", "hit_at_1"],
        },
    )
    assert response.status_code == 400
    assert "retrieval metric" in response.json()["error"]["message"]
    assert api_client.get(f"{API}/experiments/exp-retrieval-as-answer-eval").status_code == 404
