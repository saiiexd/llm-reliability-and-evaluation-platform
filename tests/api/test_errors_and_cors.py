"""
Tests for global exception handling, correlation ID middleware, and CORS policies.
"""

from fastapi.testclient import TestClient


def test_request_correlation_and_timing_headers(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert "X-Response-Time-Ms" in res.headers

    # Custom request ID preserved
    custom_id = "custom-trace-uuid-12345"
    res_custom = api_client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert res_custom.status_code == 200
    assert res_custom.headers["X-Request-ID"] == custom_id


def test_validation_error_envelope(api_client: TestClient) -> None:
    # Malformed payload (missing required fields)
    res = api_client.post("/api/v1/experiments", json={"invalid": "payload"})
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(data["error"]["details"], list)


def test_cors_preflight_and_origin(api_client: TestClient) -> None:
    res = api_client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"
