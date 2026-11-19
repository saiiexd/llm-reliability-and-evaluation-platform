"""
LLM provider discovery and readiness endpoints.

Reports provider readiness, supported models, and default parameters without
disclosing secrets or issuing paid API requests.
"""

from __future__ import annotations

from fastapi import APIRouter

from llm_reliability.api.registry import get_providers_metadata
from llm_reliability.api.schemas.providers import (
    ProviderModelInfo,
    ProvidersListResponse,
    ProviderStatusResponse,
)

router = APIRouter(prefix="/providers", tags=["Providers"])


@router.get("", response_model=ProvidersListResponse)
def list_providers() -> ProvidersListResponse:
    """List available LLM providers, readiness status, and supported models."""
    raw_metadata = get_providers_metadata()
    providers: list[ProviderStatusResponse] = []

    for item in raw_metadata:
        supported_models = [
            ProviderModelInfo(
                model_id=m["model_id"],
                display_name=m["display_name"],
                context_window=m["context_window"],
                default_temperature=m["default_temperature"],
            )
            for m in item.get("supported_models", [])
        ]
        providers.append(
            ProviderStatusResponse(
                provider_id=item["provider_id"],
                name=item["name"],
                is_configured=item["is_configured"],
                base_url=item.get("base_url"),
                requires_api_key=item.get("requires_api_key", True),
                supported_models=supported_models,
                default_model=item["default_model"],
            )
        )

    return ProvidersListResponse(providers=providers)
