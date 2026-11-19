"""
Pytest fixtures for API testing with isolated temporary persistence stores.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_reliability.api.app import create_app
from llm_reliability.api.config import ApiSettings
from llm_reliability.api.dependencies import get_settings
from scripts.seed_live_experiments import seed_store


@pytest.fixture
def temp_store() -> Generator[Path]:
    with tempfile.TemporaryDirectory(prefix="llm-api-test-store-") as tmp:
        store_path = Path(tmp)
        seed_store(store_path)
        yield store_path


@pytest.fixture
def api_client(temp_store: Path) -> TestClient:
    settings = ApiSettings(
        persistence_dir=temp_store,
        cors_origins=["http://localhost:5173", "http://testserver"],
        environment="test",
        debug=True,
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)
