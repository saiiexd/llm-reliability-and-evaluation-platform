"""
Tests for provider discovery and readiness endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_reliability.api.app import create_app
from llm_reliability.api.config import ApiSettings


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = ApiSettings(
        persistence_dir=tmp_path / "experiments",
        environment="test",
    )
    app = create_app(settings=settings)
    return TestClient(app)


def test_list_providers_endpoint(client: TestClient) -> None:
    """GET /api/v1/providers lists providers and supported models without leaking credentials."""
    resp = client.get("/api/v1/providers")
    assert resp.status_code == 200

    data = resp.json()
    assert "providers" in data
    providers = data["providers"]
    assert len(providers) >= 2

    provider_ids = [p["provider_id"] for p in providers]
    assert "openai" in provider_ids
    assert "openai-compatible" in provider_ids

    openai_provider = next(p for p in providers if p["provider_id"] == "openai")
    assert openai_provider["name"] == "OpenAI"
    assert isinstance(openai_provider["is_configured"], bool)
    assert openai_provider["requires_api_key"] is True
    assert len(openai_provider["supported_models"]) >= 2
    model_ids = [m["model_id"] for m in openai_provider["supported_models"]]
    assert "gpt-4o-mini" in model_ids
    assert "gpt-4o" in model_ids

    # Critical security check: No API keys or tokens in response
    resp_text = resp.text.lower()
    assert "sk-" not in resp_text
    assert "api_key" not in resp.json()["providers"][0]  # Only 'requires_api_key'
