"""
Tests for robustness perturbation comparisons and consistency evaluations.
"""

from fastapi.testclient import TestClient


def test_list_robustness_and_filtering(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/robustness?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 1
    comp = data["items"][0]
    assert comp["baseline_test_case_id"] == "correct-context-correct-answer"
    assert comp["perturbed_test_case_id"] == "correct-context-correct-answer-reworded"
    assert comp["perturbation_type"] == "question_rewording"


def test_get_run_robustness_bundle(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/robustness/run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == "run-rag-baseline-001"
    assert len(data["comparisons"]) == 1
    assert data["consistency"] is not None
    assert data["consistency"]["total_comparisons"] == 1
    assert data["consistency"]["consistency_rate"] == 1.0


def test_get_robustness_bundle_not_found(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/robustness/run-unknown-999")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
