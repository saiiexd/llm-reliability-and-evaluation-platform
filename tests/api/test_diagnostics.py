"""
Tests for reliability diagnostics and forensic claim assessments.
"""

from fastapi.testclient import TestClient


def test_list_diagnostics_and_filtering(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/diagnostics?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 7

    # Filter by failure_category
    res_correct = api_client.get(
        "/api/v1/diagnostics?run_id=run-rag-baseline-001&failure_category=correct"
    )
    assert res_correct.status_code == 200
    for d in res_correct.json()["items"]:
        assert d["failure_category"] == "correct"

    # Filter by secondary finding
    res_sec = api_client.get(
        "/api/v1/diagnostics?run_id=run-rag-baseline-001&secondary_finding=unsupported"
    )
    assert res_sec.status_code == 200
    for d in res_sec.json()["items"]:
        assert any("unsupported" in f.lower() for f in d["secondary_findings"])


def test_get_individual_diagnostic_with_claims(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/diagnostics/run-rag-baseline-001/correct-context-correct-answer")
    assert res.status_code == 200
    data = res.json()
    assert data["test_case_id"] == "correct-context-correct-answer"
    assert data["failure_category"] == "correct"
    assert data["question"] == "What is the capital of France?"
    assert data["generated_answer"] == "Paris"
    assert isinstance(data["claim_assessments"], list)
    assert len(data["claim_assessments"]) > 0
    assert data["claim_assessments"][0]["statement"] == "Paris"


def test_get_diagnostic_not_found(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/diagnostics/run-rag-baseline-001/tc-unknown-999")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
