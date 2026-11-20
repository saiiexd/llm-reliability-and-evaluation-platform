"""
Tests for controlled experiment execution, adapter allowlisting, and security boundaries.
"""

from fastapi.testclient import TestClient


def test_controlled_execution_success(api_client: TestClient) -> None:
    res = api_client.post(
        "/api/v1/experiments/exp-baseline-non-rag-demo/execute",
        json={"adapter_id": "mock-adapter-v1", "run_id": "run-api-exec-test-01"},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["run_id"] == "run-api-exec-test-01"
    assert data["experiment_id"] == "exp-baseline-non-rag-demo"
    assert data["status"] == "succeeded"
    assert data["num_succeeded"] == 2
    assert data["num_failed"] == 0

    # Verify run is immediately queryable via GET /runs/{run_id}
    run_res = api_client.get("/api/v1/runs/run-api-exec-test-01")
    assert run_res.status_code == 200
    assert run_res.json()["run_id"] == "run-api-exec-test-01"


def test_controlled_execution_rejects_unallowlisted_adapter(
    api_client: TestClient,
) -> None:
    # Attempting to request an unallowlisted or arbitrary adapter
    res = api_client.post(
        "/api/v1/experiments/exp-baseline-non-rag-demo/execute",
        json={"adapter_id": "os_system_shell_adapter"},
    )
    assert res.status_code == 400
    data = res.json()
    assert "Unknown adapter" in data["error"]["message"]


def test_controlled_execution_rejects_arbitrary_python_paths(
    api_client: TestClient,
) -> None:
    res = api_client.post(
        "/api/v1/experiments/exp-baseline-non-rag-demo/execute",
        json={"adapter_id": "../../etc/passwd; rm -rf /"},
    )
    assert res.status_code == 400
    assert "Unknown adapter" in res.json()["error"]["message"]


def test_execute_nonexistent_experiment(api_client: TestClient) -> None:
    res = api_client.post(
        "/api/v1/experiments/exp-ghost-999/execute",
        json={"adapter_id": "mock-adapter-v1"},
    )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
