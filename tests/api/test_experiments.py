"""
Tests for experiment listing, detail, creation, and configuration safety.
"""

from fastapi.testclient import TestClient


def test_list_experiments_and_pagination(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/experiments?limit=2&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 2
    assert data["pagination"]["total"] >= 3
    assert data["pagination"]["has_next"] is True
    assert data["pagination"]["has_previous"] is False


def test_search_experiments(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/experiments?search=console-demo")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["experiment_id"] == "exp-reliability-console-demo"


def test_get_experiment_detail_and_redaction(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/experiments/exp-reliability-console-demo")
    assert res.status_code == 200
    data = res.json()
    assert data["experiment_id"] == "exp-reliability-console-demo"
    assert data["runs_count"] >= 2
    assert len(data["evaluator_configs"]) >= 2
    assert "config_fingerprint" in data


def test_get_experiment_not_found(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/experiments/exp-does-not-exist")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_list_experiment_runs(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/experiments/exp-reliability-console-demo/runs")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) >= 2
    for r in data["items"]:
        assert r["experiment_id"] == "exp-reliability-console-demo"


def test_create_experiment_success_and_conflict(api_client: TestClient) -> None:
    payload = {
        "experiment_id": "exp-custom-new-1",
        "name": "Custom Test Experiment",
        "description": "Test experiment created via API",
        "dataset_name": "custom-dataset",
        "test_cases": [
            {
                "id": "c1",
                "input": "What is 2+2?",
                "reference_answer": "4",
                "metadata": {},
            }
        ],
        "model_config": {
            "model_id": "mock-adapter-v1",
            "temperature": 0.0,
            "metadata": {"api_key": "super-secret-token-12345"},
        },
        "evaluator_names": ["exact_match"],
        "metadata": {"secret_token": "hidden-secret"},
    }

    # Credentials must never be persisted in an experiment record: rejected.
    secret_res = api_client.post("/api/v1/experiments", json=payload)
    assert secret_res.status_code == 422
    body = secret_res.json()["error"]
    assert body["code"] == "CONFIGURATION_SECURITY_ERROR"
    assert sorted(body["details"]["credential_keys"]) == [
        "metadata.secret_token",
        "model_config.metadata.api_key",
    ]
    assert "super-secret" not in secret_res.text
    assert api_client.get("/api/v1/experiments/exp-custom-new-1").status_code == 404

    payload["model_config"]["metadata"] = {"max_tokens_note": "harmless"}
    payload["metadata"] = {"owner": "team"}
    create_res = api_client.post("/api/v1/experiments", json=payload)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["experiment_id"] == "exp-custom-new-1"

    # Verify duplicate creation triggers 409 Conflict
    conflict_res = api_client.post("/api/v1/experiments", json=payload)
    assert conflict_res.status_code == 409
    assert conflict_res.json()["error"]["code"] == "RESOURCE_CONFLICT"


def test_create_experiment_unknown_evaluator_rejected(api_client: TestClient) -> None:
    payload = {
        "experiment_id": "exp-bad-evaluator",
        "name": "Bad Evaluator Experiment",
        "dataset_name": "dataset",
        "test_cases": [{"id": "c1", "input": "Q", "reference_answer": "A"}],
        "model_config": {"model_id": "mock-adapter-v1"},
        "evaluator_names": ["unauthorized_evaluator_xyz"],
    }
    res = api_client.post("/api/v1/experiments", json=payload)
    assert res.status_code == 400
    assert "Unknown evaluator" in res.json()["error"]["message"]
