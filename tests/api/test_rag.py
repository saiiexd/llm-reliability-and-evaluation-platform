"""
Tests for RAG retrieval results, retrieved chunks, and evaluation metrics.
"""

from fastapi.testclient import TestClient


def test_list_retrieval_results(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/rag/retrieval-results?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 7


def test_get_individual_retrieval_result(api_client: TestClient) -> None:
    res = api_client.get(
        "/api/v1/rag/retrieval-results/run-rag-baseline-001/correct-context-correct-answer"
    )
    assert res.status_code == 200
    data = res.json()
    assert data["test_case_id"] == "correct-context-correct-answer"
    assert len(data["retrieved_chunks"]) >= 1
    chunk = data["retrieved_chunks"][0]
    assert "Paris" in chunk["text"]
    assert chunk["rank"] == 1
    assert chunk["similarity_score"] is not None

    # Verify retrieval evaluations contain Hit@1, Recall@1, MRR
    assert len(data["retrieval_evaluations"]) >= 3
    metric_names = [e["metric_name"] for e in data["retrieval_evaluations"]]
    assert "hit_at_k" in metric_names
    assert "mrr" in metric_names


def test_get_retrieval_result_not_found(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/rag/retrieval-results/run-rag-baseline-001/tc-unknown")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
