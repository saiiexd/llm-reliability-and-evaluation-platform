"""
Pydantic schemas for LLM provider status and capabilities.
"""

from __future__ import annotations

from llm_reliability.api.schemas.common import BaseApiSchema


class ProviderModelInfo(BaseApiSchema):
    model_id: str
    display_name: str
    context_window: int
    default_temperature: float


class ProviderStatusResponse(BaseApiSchema):
    provider_id: str
    name: str
    is_configured: bool
    base_url: str | None = None
    requires_api_key: bool
    supported_models: list[ProviderModelInfo]
    default_model: str


class ProvidersListResponse(BaseApiSchema):
    providers: list[ProviderStatusResponse]

