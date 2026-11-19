"""
Regression tests for the live API registries, exercised end to end.

The unit tests for each evaluator used hand-written fakes, so they never
noticed that the fake judge wired into the *API* registry (and the seed
script) replied with keys the strict judge parser rejects. Every faithfulness
result in the live app was therefore an ``output_validation_error`` and every
RAG case was counted as failed. These tests go through the real registries.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_reliability.api.registry import JUDGE_RESPONSES
from llm_reliability.evaluation.llm_judge import parse_judge_output
from llm_reliability.rag.faithfulness_judge import FAITHFULNESS_RUBRIC

API = "/api/v1"


def test_canned_judge_replies_conform_to_faithfulness_rubric() -> None:
    for test_case_id, raw in JUDGE_RESPONSES.items():
        parsed = parse_judge_output(raw, FAITHFULNESS_RUBRIC)
        assert parsed.outcome in FAITHFULNESS_RUBRIC.scale, test_case_id


def test_seeded_runs_have_no_judge_validation_errors(api_client: TestClient) -> None:
    run = api_client.get(f"{API}/runs/run-rag-baseline-001").json()
    judge_results = [
        e
        for tc in run["test_case_results"]
        for e in tc["evaluations"]
        if e["evaluator_name"] == "llm_faithfulness_judge"
    ]
    assert judge_results, "seeded RAG run should include faithfulness judge results"
    assert all(e["status"] == "success" for e in judge_results), judge_results
    # The correct, supported answer must now be able to pass overall.
    by_id = {tc["test_case_id"]: tc for tc in run["test_case_results"]}
    assert by_id["correct-context-correct-answer"]["passed"] is True
    assert run["num_passed"] > 0


def test_retrieval_metric_rejected_as_answer_evaluator(api_client: TestClient) -> None:
    response = api_client.post(
        f"{API}/experiments",
        json={
            "experiment_id": "exp-retrieval-as-answer-eval",
            "name": "x",
            "dataset_name": "d",
            "test_cases": [{"id": "tc-1", "input": "q", "reference_answer": "a"}],
            "model_config": {"model_id": "rag-pipeline-v1"},
            "evaluator_names": ["exact_match", "hit_at_1"],
        },
    )
    assert response.status_code == 400
    assert "retrieval metric" in response.json()["error"]["message"]
    assert api_client.get(f"{API}/experiments/exp-retrieval-as-answer-eval").status_code == 404


def test_runs_record_the_adapter_that_actually_executed(api_client: TestClient) -> None:
    """Benchmarking one experiment on two adapters must yield distinguishable runs."""
    create = api_client.post(
        f"{API}/experiments",
        json={
            "experiment_id": "exp-provenance",
            "name": "Provenance",
            "dataset_name": "d",
            "test_cases": [{"id": "tc-1", "input": "q", "reference_answer": "a"}],
            "model_config": {"model_id": "declared-label"},
            "evaluator_names": ["exact_match"],
        },
    )
    assert create.status_code == 201, create.text
    bench = api_client.post(
        f"{API}/benchmarks/execute",
        json={
            "experiment_id": "exp-provenance",
            "adapter_ids": ["echo-adapter", "mock-adapter-v1"],
        },
    )
    assert bench.status_code in (200, 201), bench.text
    runs = api_client.get(f"{API}/experiments/exp-provenance/runs").json()["items"]
    assert sorted(r["adapter_id"] for r in runs) == ["echo-adapter", "mock-adapter-v1"]
    for r in runs:
        assert r["executed_model_ids"] == [r["adapter_id"]]


def test_reference_free_case_is_accepted_and_skipped_not_failed(api_client: TestClient) -> None:
    create = api_client.post(
        f"{API}/experiments",
        json={
            "experiment_id": "exp-reference-free",
            "name": "Reference-free",
            "dataset_name": "d",
            "test_cases": [{"id": "tc-1", "input": "Say something"}],
            "model_config": {"model_id": "echo-adapter"},
            "evaluator_names": ["exact_match"],
        },
    )
    assert create.status_code == 201, create.text
    run = api_client.post(
        f"{API}/experiments/exp-reference-free/execute", json={"adapter_id": "echo-adapter"}
    ).json()
    detail = api_client.get(f"{API}/runs/{run['run_id']}").json()
    evaluation = detail["test_case_results"][0]["evaluations"][0]
    assert evaluation["status"] == "skipped"
    assert evaluation["passed"] is None


def test_seeded_runs_are_labelled_simulated(api_client: TestClient) -> None:
    run = api_client.get(f"{API}/runs/run-rag-baseline-001").json()
    joined = " ".join(run["simulated_components"])
    assert "mock adapter" in joined
    assert "fake embedding" in joined
    assert "canned judge" in joined
    summary = api_client.get(f"{API}/runs?limit=50").json()["items"]
    assert all(r["simulated_components"] for r in summary)


def test_rag_pipeline_must_match_declared_retrieval_config(api_client: TestClient) -> None:
    """A top_k=2 experiment must not be recorded from a top_k=1 pipeline."""
    wrong = api_client.post(
        f"{API}/experiments/exp-reliability-topk2-demo/execute",
        json={"adapter_id": "rag-pipeline-v1"},
    )
    assert wrong.status_code == 400
    assert "top_k" in wrong.json()["error"]["message"]

    right = api_client.post(
        f"{API}/experiments/exp-reliability-topk2-demo/execute",
        json={"adapter_id": "rag-pipeline-topk-2"},
    )
    assert right.status_code == 201, right.text


def test_agreement_kappa_and_table_reach_the_api(api_client: TestClient) -> None:
    api_client.post(
        f"{API}/experiments",
        json={
            "experiment_id": "exp-agreement",
            "name": "Agreement",
            "dataset_name": "d",
            "test_cases": [
                {"id": f"tc-{i}", "input": "same" if i < 2 else f"q{i}", "reference_answer": "same"}
                for i in range(6)
            ],
            "model_config": {"model_id": "echo-adapter"},
            "evaluator_names": ["exact_match", "squad_match"],
        },
    )
    run_id = api_client.post(
        f"{API}/experiments/exp-agreement/execute", json={"adapter_id": "echo-adapter"}
    ).json()["run_id"]
    body = api_client.get(f"{API}/analytics/evaluators", params={"run_id": run_id}).json()
    pair = body["agreements"][0]
    assert (
        pair["both_passed"],
        pair["only_a_passed"],
        pair["only_b_passed"],
        pair["both_failed"],
    ) == (2, 0, 0, 4)
    assert pair["cohens_kappa"] == 1.0
    assert pair["excluded_cases"] == 0


def test_credential_keys_in_any_casing_are_rejected(api_client: TestClient) -> None:
    from llm_reliability.api.schemas.common import find_credential_keys

    found = find_credential_keys(
        {
            "secret_key": 1,
            "aws_secret_access_key": 1,
            "openaiApiKey": 1,
            "private-key": 1,
            "token": 1,
            "max_tokens": 1,
            "top_k": 1,
            "tokenizer": 1,
        }
    )
    assert sorted(found) == [
        "aws_secret_access_key",
        "openaiApiKey",
        "private-key",
        "secret_key",
        "token",
    ]
