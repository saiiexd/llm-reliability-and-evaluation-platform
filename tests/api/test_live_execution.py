"""
Real-model execution path through the API, with the model server faked.

Covers: dataset presets, server-filled RAG / prompt-template configuration,
real-judge definitions, the external-inference gate for judges, adapter and
template guards, background jobs, and that a run built this way is not
labelled simulated. No network: ``openai_compatible_provider`` and the
embedder are replaced with deterministic fakes.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_reliability.api import live
from llm_reliability.datasets import load_squad_subset
from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse
from llm_reliability.experiments.models import DatasetVersion
from llm_reliability.rag.embeddings import EmbeddingModel

PRESET = "squad-v1.1-dev-subset-v1"
REAL_PROVIDER_FACTORY = live.openai_compatible_provider  # captured before fixtures patch it
MODEL = "qwen3:4b-instruct"


class FakeServerModel(ModelAdapter):
    """Answers like a model server: judge prompts get rubric JSON, others an answer."""

    calls: list[str] = []

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        FakeServerModel.calls.append(self.model_id)
        prompt = request.input_text
        if "relative to a provided reference answer" in prompt:
            text = json.dumps({"outcome": "correct", "reasoning": "Matches the reference."})
        elif "faithful to a provided set of retrieved context" in prompt:
            text = json.dumps({"outcome": "supported", "reasoning": "Stated in passage 1."})
        else:
            text = "I don't know"
        return ModelResponse(
            test_case_id=request.test_case_id,
            output_text=text,
            model_id=self.model_id,
            latency_ms=1.0,
        )


class HashEmbedding(EmbeddingModel):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:8]]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def fake_model_server(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeServerModel.calls = []
    monkeypatch.setenv("LLM_RELIABILITY_OPENAI_BASE_URL", "http://localhost:9/v1")

    def provider(model_id: str, allow_external_inference: bool, client: Any = None) -> Any:
        return FakeServerModel(model_id)

    def embedder(model_id: str, allow_external_inference: bool, client: Any = None) -> Any:
        if not allow_external_inference:
            raise ValueError("opt-in required")
        return HashEmbedding(model_id)

    monkeypatch.setattr(live, "openai_compatible_provider", provider)
    monkeypatch.setattr(live, "cached_embedder", embedder)


def _create(client: TestClient, **overrides: Any) -> Any:
    body: dict[str, Any] = {
        "experiment_id": "live-rag",
        "name": "live",
        "dataset_preset": PRESET,
        "dataset_limit": 3,
        "retrieval_top_k": 2,
        "judge_model_id": MODEL,
        "model_config": {"model_id": MODEL, "temperature": 0.0, "max_tokens": 64},
        "evaluator_names": ["squad_match", "llm_judge", "llm_faithfulness_judge"],
    }
    body.update(overrides)
    return client.post("/api/v1/experiments", json=body)


def test_dataset_presets_are_listed(api_client: TestClient) -> None:
    presets = api_client.get("/api/v1/datasets").json()
    assert [p["preset_id"] for p in presets] == [PRESET]
    assert presets[0]["test_case_count"] == 60
    assert presets[0]["has_retrieval_labels"] is True
    assert presets[0]["license"] == "CC BY-SA 4.0"


def test_preset_dataset_is_identical_to_the_study_dataset(api_client: TestClient) -> None:
    res = _create(api_client, dataset_limit=None)
    assert res.status_code == 201, res.json()
    expected = DatasetVersion(dataset=load_squad_subset().dataset()).version_id
    assert res.json()["dataset_version"]["version_id"] == expected


def test_rag_config_is_filled_by_the_server(api_client: TestClient) -> None:
    detail = _create(api_client).json()
    rag = detail["model_config"]["extra_params"]["rag"]
    assert rag["top_k"] == 2
    assert rag["embedding_model_id"] == "nomic-embed-text"
    assert rag["corpus_fingerprint"] == load_squad_subset().corpus_fingerprint()
    assert rag["prompt_template_id"] == "concise_v1"
    judge = next(e for e in detail["evaluator_configs"] if e["evaluator_name"] == "llm_judge")
    assert judge["parameters"]["judge_model_id"] == MODEL


def test_closed_book_preset_declares_its_prompt_template(api_client: TestClient) -> None:
    detail = _create(api_client, retrieval_top_k=None).json()
    extra = detail["model_config"]["extra_params"]
    assert "rag" not in extra
    assert extra["prompt_template_id"] == "closed_book_concise_v1"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"test_cases": [{"id": "a", "input": "q"}], "dataset_name": "d"}, "exactly one"),
        ({"dataset_preset": None, "test_cases": None}, "exactly one"),
        (
            {
                "dataset_preset": None,
                "dataset_name": "d",
                "test_cases": [{"id": "a", "input": "q"}],
            },
            "apply only to a 'dataset_preset'",
        ),
        ({"dataset_preset": "unknown-set"}, "Unknown dataset preset"),
    ],
)
def test_invalid_dataset_requests_are_rejected(
    api_client: TestClient, overrides: dict[str, Any], message: str
) -> None:
    res = _create(api_client, **overrides)
    assert res.status_code in (400, 422)
    assert message in json.dumps(res.json())


def test_correctness_judge_needs_a_judge_model(api_client: TestClient) -> None:
    res = _create(api_client, judge_model_id=None)
    assert res.status_code == 400
    assert "needs a judge model" in res.json()["error"]["message"]


def test_live_judges_require_opt_in_even_with_an_offline_adapter(
    api_client: TestClient,
) -> None:
    _create(api_client, retrieval_top_k=None, evaluator_names=["llm_judge"])
    res = api_client.post(
        "/api/v1/experiments/live-rag/execute", json={"adapter_id": "openai-custom"}
    )
    assert res.status_code == 400
    assert "evaluator 'llm_judge' (judge qwen3:4b-instruct)" in res.json()["error"]["message"]
    assert FakeServerModel.calls == []


def test_rag_experiment_is_refused_by_a_non_rag_adapter(api_client: TestClient) -> None:
    _create(api_client)
    res = api_client.post(
        "/api/v1/experiments/live-rag/execute",
        json={"adapter_id": "openai-custom", "allow_external_inference": True},
    )
    assert res.status_code == 400
    assert "performs no retrieval" in res.json()["error"]["message"]


def test_closed_book_template_is_enforced(api_client: TestClient) -> None:
    _create(api_client, retrieval_top_k=None, evaluator_names=["squad_match"])
    res = api_client.post(
        "/api/v1/experiments/live-rag/execute", json={"adapter_id": "mock-adapter-v1"}
    )
    assert res.status_code == 400
    assert "declares prompt template 'closed_book_concise_v1'" in res.json()["error"]["message"]


def test_live_rag_run_records_real_provenance(api_client: TestClient) -> None:
    _create(api_client)
    res = api_client.post(
        "/api/v1/experiments/live-rag/execute",
        json={"adapter_id": "openai-custom-rag", "allow_external_inference": True},
    )
    assert res.status_code == 201, res.json()
    run = api_client.get(f"/api/v1/runs/{res.json()['run_id']}").json()
    assert run["simulated_components"] == []
    assert run["executed_model_ids"] == [MODEL]
    assert run["adapter_id"] == "openai-custom-rag"
    statuses = {
        e["evaluator_name"]: e["status"]
        for tc in run["test_case_results"]
        for e in tc["evaluations"]
    }
    assert statuses == {
        "squad_match": "success",
        "llm_judge": "success",
        "llm_faithfulness_judge": "success",
    }
    retrieval = api_client.get(
        "/api/v1/rag/retrieval-results", params={"run_id": run["run_id"]}
    ).json()
    metrics = {
        r["metric_name"] for item in retrieval["items"] for r in item["retrieval_evaluations"]
    }
    assert {"hit_at_k", "recall_at_k", "precision_at_k", "mrr"} <= metrics


def test_background_job_reports_progress_and_the_persisted_run(api_client: TestClient) -> None:
    _create(api_client)
    res = api_client.post(
        "/api/v1/experiments/live-rag/jobs",
        json={"adapter_id": "openai-custom-rag", "allow_external_inference": True},
    )
    assert res.status_code == 202, res.json()
    job_id = res.json()["job_id"]
    assert res.json()["total"] == 3
    deadline = time.monotonic() + 10
    job = res.json()
    while job["status"] not in ("succeeded", "failed") and time.monotonic() < deadline:
        time.sleep(0.05)
        job = api_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "succeeded", job
    assert job["completed"] == 3
    assert api_client.get(f"/api/v1/runs/{job['run_id']}").status_code == 200
    assert any(j["job_id"] == job_id for j in api_client.get("/api/v1/jobs").json())


def test_job_validation_errors_are_immediate(api_client: TestClient) -> None:
    _create(api_client)
    res = api_client.post(
        "/api/v1/experiments/live-rag/jobs", json={"adapter_id": "openai-custom-rag"}
    )
    assert res.status_code == 400
    assert api_client.get("/api/v1/jobs/job-does-not-exist").status_code == 404


def test_declared_judge_with_a_changed_rubric_cannot_be_reproduced() -> None:
    declared = {
        "judge_model_id": MODEL,
        "judge_model_config": {"model_id": MODEL, "temperature": 0.0},
        "rubric": {"rubric_id": "something_else_v9"},
    }
    with pytest.raises(ValueError, match="cannot be reproduced"):
        live.build_live_judge("llm_judge", True, declared=declared)


def test_judge_experiment_can_be_defined_without_a_configured_server(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use the real provider factory: defining must not need a server or key.
    monkeypatch.setattr(live, "openai_compatible_provider", REAL_PROVIDER_FACTORY)
    for var in (
        "LLM_RELIABILITY_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_RELIABILITY_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    res = _create(api_client, judge_rubric_version="v2")
    assert res.status_code == 201, res.json()
    rubrics = {
        e["evaluator_name"]: e["parameters"]["rubric"]["rubric_id"]
        for e in res.json()["evaluator_configs"]
        if "rubric" in e["parameters"]
    }
    assert rubrics == {
        "llm_judge": "correctness_vs_reference_v2",
        "llm_faithfulness_judge": "faithfulness_vs_context_v2",
    }
