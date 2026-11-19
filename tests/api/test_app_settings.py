"""
Regression test: an app built with explicit settings must use them everywhere.

Previously ``create_app(settings)`` configured CORS and docs from ``settings``
but route dependencies still resolved the cached process-wide
``get_settings()``, so experiments created through such an app were written
to the default ``./experiments`` store instead of ``settings.persistence_dir``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llm_reliability.api.app import create_app
from llm_reliability.api.config import ApiSettings


def test_create_app_routes_use_supplied_persistence_dir(tmp_path: Path) -> None:
    store = tmp_path / "store"
    app = create_app(ApiSettings(persistence_dir=store, environment="test"))
    client = TestClient(app)

    response = client.post(
        "/api/v1/experiments",
        json={
            "experiment_id": "exp-settings-check",
            "name": "Settings check",
            "dataset_name": "d",
            "test_cases": [{"id": "tc-1", "input": "q", "reference_answer": "a"}],
            "model_config": {"model_id": "mock-adapter-v1"},
            "evaluator_names": ["exact_match"],
        },
    )

    assert response.status_code == 201, response.text
    assert (store / "experiments" / "exp-settings-check.json").is_file()
