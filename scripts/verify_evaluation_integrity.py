"""
Verification script for Prompt 10: Evaluation Integrity Audit & Pass-Rate Consistency.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from llm_reliability.api.app import create_app
from llm_reliability.api.config import ApiSettings


def main() -> None:
    # Runs against a throwaway store so the audit never writes into ./experiments.
    store = Path(tempfile.mkdtemp(prefix="llm-integrity-audit-"))
    client = TestClient(create_app(ApiSettings(persistence_dir=store)))

    exp_id = "exp-geo-audit-test"

    # 1. Create geography baseline experiment
    exp_payload = {
        "experiment_id": exp_id,
        "name": "Geography Baseline Integrity Audit",
        "description": "Audit test for Prompt 10 pass-rate consistency",
        "dataset_name": "qa-geography-baseline-v1",
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
            {
                "id": "tc-3",
                "input": "What is the largest ocean on Earth?",
                "reference_answer": "Pacific Ocean",
                "metadata": {},
            },
            {
                "id": "tc-4",
                "input": "Which element has atomic number 1?",
                "reference_answer": "Hydrogen",
                "metadata": {},
            },
            {
                "id": "tc-5",
                "input": "What year did the Apollo 11 mission land on the Moon?",
                "reference_answer": "1969",
                "metadata": {},
            },
        ],
        "model_config": {"model_id": "mock-adapter-v1", "temperature": 0.0, "max_tokens": 100},
        "evaluator_names": ["exact_match"],
        "metadata": {},
    }

    create_res = client.post("/api/v1/experiments", json=exp_payload)
    assert create_res.status_code == 201, f"Failed create: {create_res.text}"
    exp_data = create_res.json()
    test_case_count = exp_data["dataset_version"]["test_case_count"]
    print(f"[1] Experiment created. Dataset test_case_count in definition: {test_case_count}")
    assert test_case_count == 5, f"Expected 5 test cases in definition, got {test_case_count}"

    # 2. Check experiment list
    list_res = client.get("/api/v1/experiments?search=Geography")
    assert list_res.status_code == 200
    matched = [item for item in list_res.json()["items"] if item["experiment_id"] == exp_id]
    assert len(matched) == 1
    summary_count = matched[0]["test_case_count"]
    print(f"[2] Experiment summary test_case_count: {summary_count}")
    assert summary_count == 5

    # 3. Execute experiment
    exec_res = client.post(
        f"/api/v1/experiments/{exp_id}/execute", json={"adapter_id": "mock-adapter-v1"}
    )
    assert exec_res.status_code == 201, f"Execution failed: {exec_res.text}"
    exec_data = exec_res.json()
    print(
        f"[3] Execution result: status={exec_data['status']}, "
        f"num_succeeded={exec_data['num_succeeded']}, "
        f"num_passed={exec_data['num_passed']}, "
        f"pass_rate={exec_data['pass_rate']}"
    )
    assert exec_data["num_succeeded"] == 5, "All 5 should have succeeded execution"
    assert exec_data["num_passed"] == 0, "0 should have passed evaluation (exact match failed)"
    assert exec_data["pass_rate"] == 0.0, "Pass rate must be 0.0%"

    run_id = exec_data["run_id"]

    # 4. Check run detail
    run_res = client.get(f"/api/v1/runs/{run_id}")
    assert run_res.status_code == 200
    run_data = run_res.json()
    print(
        f"[4] Run detail: num_test_cases={run_data['num_test_cases']}, "
        f"num_succeeded={run_data['num_succeeded']}, "
        f"num_failed={run_data['num_failed']}, "
        f"num_passed={run_data['num_passed']}, "
        f"pass_rate={run_data['pass_rate']}"
    )
    assert run_data["num_test_cases"] == 5
    assert run_data["num_succeeded"] == 5
    assert run_data["num_failed"] == 0
    assert run_data["num_passed"] == 0
    assert run_data["pass_rate"] == 0.0

    # 5. Check test case results
    for tc in run_data["test_case_results"]:
        eval_first = tc["evaluations"][0]
        tc_id = tc["test_case_id"]
        succeeded = tc["succeeded"]
        passed = tc["passed"]
        eval_passed = eval_first["passed"]
        eval_score = eval_first["score"]
        print(
            f"[5] Case {tc_id}: succeeded={succeeded}, passed={passed}, "
            f"evaluator_verdict={eval_passed}, score={eval_score}"
        )
        assert succeeded is True, f"Case {tc_id} generation should succeed"
        assert passed is False, f"Case {tc_id} evaluation should fail"
        assert eval_passed is False, f"Case {tc_id} exact_match verdict should be False"
        assert eval_score == 0.0

    # 6. Check filtering by passed=true and passed=false
    pass_filter_res = client.get(f"/api/v1/runs/{run_id}/test-cases?passed=true")
    assert pass_filter_res.status_code == 200
    pass_items = pass_filter_res.json()["items"]
    assert len(pass_items) == 0, f"Expected 0 passed items, got {len(pass_items)}"

    fail_filter_res = client.get(f"/api/v1/runs/{run_id}/test-cases?passed=false")
    assert fail_filter_res.status_code == 200
    fail_items = fail_filter_res.json()["items"]
    assert len(fail_items) == 5, f"Expected 5 failed items, got {len(fail_items)}"

    print("\nSUCCESS: All geography baseline evaluation integrity audit checks passed flawlessly!")


if __name__ == "__main__":
    main()
