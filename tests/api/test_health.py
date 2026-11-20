"""
Tests for API health and readiness endpoints.
"""

from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["total_experiments"] >= 3
    assert data["total_runs"] >= 4
    assert data["environment"] == "test"
    assert "timestamp" in data
    # Ensure no secret or internal directory paths are leaked in health
    assert "api_key" not in str(data).lower()
