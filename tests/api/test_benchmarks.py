"""
Tests for benchmark executions and multi-model repetition semantics.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_reliability.api.app import create_app
from llm_reliability.api.config import ApiSettings


@pytest.fixture
def test_setup(tmp_path):
    persistence_dir = tmp_path / "experiments"
    settings = ApiSettings(
        persistence_dir=persistence_dir,
        environment="test",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    import uuid

    exp_id = f"exp-{uuid.uuid4().hex[:8]}"

    # Create a baseline test experiment
    create_payload = {
        "experiment_id": exp_id,
        "name": "Benchmark Evaluation Experiment",
        "dataset_name": "benchmark-dataset",
        "test_cases": [
            {
                "id": "tc-1",
                "input": "What is the capital of France?",
                "reference_answer": "Paris",
                "metadata": {},
            },
            {
                "id": "tc-2",
                "input": "Bananas are rich in which mineral?",
                "reference_answer": "Potassium",
                "metadata": {},
            },
        ],
        "model_config": {
            "model_id": "test-model-v1",
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "evaluator_names": ["exact_match"],
        "metadata": {},
    }
    resp = client.post("/api/v1/experiments", json=create_payload)
    assert resp.status_code == 201

    return {"client": client, "experiment_id": exp_id}


def test_controlled_benchmark_multi_adapter_and_repetitions(test_setup) -> None:
    """Benchmark runs multiple adapters with repetitions, preserving distinct runs."""
    client: TestClient = test_setup["client"]
    exp_id: str = test_setup["experiment_id"]

    bench_req = {
        "experiment_id": exp_id,
        "adapter_ids": ["mock-adapter-v1", "echo-adapter"],
        "repetitions": 2,
        "allow_external_inference": False,
    }

    resp = client.post("/api/v1/benchmarks/execute", json=bench_req)
    assert resp.status_code == 201

    data = resp.json()
    benchmark_id = data["benchmark_id"]
    assert benchmark_id.startswith("bench-")
    assert data["total_runs"] == 4  # 2 adapters * 2 repetitions
    assert len(data["runs"]) == 4

    # Fetch runs via benchmark endpoint
    runs_resp = client.get(f"/api/v1/benchmarks/{benchmark_id}/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()
    assert len(runs) == 4

    # Verify run IDs are all distinct
    run_ids = [r["run_id"] for r in runs]
    assert len(set(run_ids)) == 4


def test_benchmark_external_inference_guard_blocks_unauthorized_external_model(test_setup) -> None:
    """Benchmarking an external provider adapter without allow_external_inference is rejected."""
    client: TestClient = test_setup["client"]
    exp_id: str = test_setup["experiment_id"]

    bench_req = {
        "experiment_id": exp_id,
        "adapter_ids": ["openai-gpt-4o-mini"],
        "repetitions": 1,
        "allow_external_inference": False,  # Not opted in!
    }

    resp = client.post("/api/v1/benchmarks/execute", json=bench_req)
    assert resp.status_code == 400
    err_msg = resp.json()["error"]["message"]
    assert "requires external inference" in err_msg
    assert "allow_external_inference=True" in err_msg


def test_single_experiment_execute_external_inference_guard(test_setup) -> None:
    """POST /api/v1/experiments/{id}/execute also rejects external provider if not opted in."""
    client: TestClient = test_setup["client"]
    exp_id: str = test_setup["experiment_id"]

    exec_req = {
        "adapter_id": "openai-gpt-4o-mini",
        "allow_external_inference": False,
    }

    resp = client.post(f"/api/v1/experiments/{exp_id}/execute", json=exec_req)
    assert resp.status_code == 400
    err_msg = resp.json()["error"]["message"]
    assert "requires external inference" in err_msg
