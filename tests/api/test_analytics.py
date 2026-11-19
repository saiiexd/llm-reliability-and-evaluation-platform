"""
Tests for cross-experiment analytics, agreement, comparison, and trends.
"""

from fastapi.testclient import TestClient


def test_analytics_overview(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_experiments"] >= 3
    assert data["total_runs"] >= 4
    assert data["total_test_cases"] > 0
    assert "exact_match" in data["evaluator_coverage"]


def test_evaluator_analytics(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/evaluators?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == "run-rag-baseline-001"
    assert "exact_match" in data["aggregates"]
    em = data["aggregates"]["exact_match"]
    assert em["num_evaluated"] == 7
    assert em["pass_rate_denominator"] == 7


def test_failure_analytics(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/failures?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert data["total_cases"] == 7
    assert "primary_categories" in data
    assert "secondary_findings" in data


def test_retrieval_analytics(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/retrieval?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3
    hit1 = next((m for m in data if m["metric_name"] == "hit_at_k" and m["k"] == 1), None)
    assert hit1 is not None
    assert hit1["evaluated_cases"] == 6
    assert hit1["skipped_cases"] == 1


def test_performance_analytics_unavailable_telemetry(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/performance?run_id=run-rag-baseline-001")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == "run-rag-baseline-001"
    assert data["tokens_available"] is False
    assert data["cost_available"] is False
    assert len(data["notes"]) > 0


def test_run_comparison_identical_and_incompatible(api_client: TestClient) -> None:
    # 1. Compare identical runs (stability)
    res_identical = api_client.get(
        "/api/v1/analytics/comparisons?run_ids=run-rag-baseline-001&run_ids=run-rag-baseline-002"
    )
    assert res_identical.status_code == 200
    identical_data = res_identical.json()
    assert identical_data["compatibility"] == "identical_configuration"
    assert len(identical_data["source_run_ids"]) == 2

    # 2. Compare materially incompatible runs (different datasets)
    res_incompat = api_client.get(
        "/api/v1/analytics/comparisons?run_ids=run-rag-baseline-001&run_ids=run-non-rag-001"
    )
    assert res_incompat.status_code == 200
    incompat_data = res_incompat.json()
    assert incompat_data["compatibility"] == "materially_incompatible"


def test_longitudinal_trends(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/trends/exp-reliability-console-demo")
    assert res.status_code == 200
    data = res.json()
    assert data["experiment_id"] == "exp-reliability-console-demo"
    assert data["has_trend"] is True
    assert len(data["run_ids"]) >= 2
    assert "exact_match_pass_rate" in data["metric_series"]


def test_repeat_consistency_groups_true_repeats(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/consistency")
    assert res.status_code == 200
    data = res.json()
    assert data["experiment_id"] is None
    groups = {g["experiment_id"]: g for g in data["groups"]}
    # The seeded store has one 2-run repeat group; every other experiment has a single run.
    assert list(groups) == ["exp-reliability-console-demo"]
    g = groups["exp-reliability-console-demo"]
    assert g["num_runs"] == 2
    assert sorted(g["run_ids"]) == ["run-rag-baseline-001", "run-rag-baseline-002"]
    assert g["adapter_recorded"] is True
    assert g["identical_output_cases"] == g["compared_cases"] == 7
    assert g["simulated_components"], "seeded mock runs must stay labelled as simulated"
    names = {e["evaluator_name"] for e in g["evaluators"]}
    assert "exact_match" in names
    for e in g["evaluators"]:
        assert e["compared_cases"] + e["excluded_cases"] == g["compared_cases"]


def test_repeat_consistency_filters_by_experiment(api_client: TestClient) -> None:
    res = api_client.get("/api/v1/analytics/consistency?experiment_id=exp-reliability-topk2-demo")
    assert res.status_code == 200
    assert res.json() == {"experiment_id": "exp-reliability-topk2-demo", "groups": []}

    missing = api_client.get("/api/v1/analytics/consistency?experiment_id=does-not-exist")
    assert missing.status_code == 404
