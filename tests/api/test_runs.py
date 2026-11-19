"""
Tests for run inspection, test cases, and execution summaries.
"""

from fastapi.testclient import TestClient


def test_list_runs_and_filtering(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/runs")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) >= 4

    # Filter by experiment
    res_filtered = api_client.get("/api/v1/runs?experiment_id=exp-baseline-non-rag-demo")
    assert res_filtered.status_code == 200
    filtered_items = res_filtered.json()["items"]
    assert len(filtered_items) == 1
    assert filtered_items[0]["run_id"] == "run-non-rag-001"


def test_get_run_detail_and_omission(api_client: TestClient) -> None:
    res_full = api_client.get("/api/v1/runs/run-rag-baseline-001")
    assert res_full.status_code == 200
    full_data = res_full.json()
    assert full_data["run_id"] == "run-rag-baseline-001"
    assert len(full_data["test_case_results"]) > 0

    # Test omitting test cases for lightweight inspection
    res_lean = api_client.get("/api/v1/runs/run-rag-baseline-001?include_test_cases=false")
    assert res_lean.status_code == 200
    lean_data = res_lean.json()
    assert len(lean_data["test_case_results"]) == 0
    assert lean_data["num_test_cases"] == full_data["num_test_cases"]


def test_get_run_not_found(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/runs/run-does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_list_run_test_cases_and_filter(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/runs/run-rag-baseline-001/test-cases?passed=true")
    assert res.status_code == 200
    data = res.json()
    for tc in data["items"]:
        assert tc["passed"] is True


def test_get_individual_test_case(api_client: TestClient) -> None:
    res = api_client.get(
        "/api/v1/runs/run-rag-baseline-001/test-cases/correct-context-correct-answer"
    )
    assert res.status_code == 200
    data = res.json()
    assert data["test_case_id"] == "correct-context-correct-answer"
    assert data["output_text"] == "Paris"
    assert len(data["evaluations"]) >= 2


def test_get_run_summaries(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/runs/run-rag-baseline-001/summaries")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == "run-rag-baseline-001"
    assert "evaluation_summary" in data
    assert "reliability_findings" in data
    assert "retrieval_summary" in data
