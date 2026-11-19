"""
Common schemas, pagination envelopes, and redaction helpers.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SECRET_PATTERN = re.compile(r"(key|secret|token|password|auth|credential|bearer)", re.IGNORECASE)


def redact_sensitive_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redacts sensitive keys such as api_key, tokens, or credentials."""
    redacted: dict[str, Any] = {}
    for k, v in data.items():
        if _SECRET_PATTERN.search(k):
            redacted[k] = "***REDACTED***"
        elif isinstance(v, dict):
            redacted[k] = redact_sensitive_dict(v)
        elif isinstance(v, list):
            redacted[k] = [
                redact_sensitive_dict(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            redacted[k] = v
    return redacted


class BaseApiSchema(BaseModel):
    """Base schema disabling protected namespace collision and enabling by-name population."""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class HealthResponse(BaseApiSchema):
    """Health readiness and service metadata."""

    status: str = Field(default="healthy", description="Application service status.")
    version: str = Field(default="0.1.0", description="Backend application version.")
    timestamp: str = Field(description="Current server UTC timestamp.")
    total_experiments: int = Field(description="Number of stored experiment definitions.")
    total_runs: int = Field(description="Number of stored experiment runs.")
    environment: str = Field(description="Server environment.")


class PaginationMeta(BaseModel):
    """Collection pagination metadata."""

    total: int = Field(description="Total number of items available.")
    limit: int = Field(description="Requested limit.")
    offset: int = Field(description="Requested offset.")
    has_next: bool = Field(description="Whether more items exist after this page.")
    has_previous: bool = Field(description="Whether items exist before this page.")


class PaginatedResponse[T](BaseModel):
    """Generic wrapper for paginated collections."""

    items: list[T]
    pagination: PaginationMeta
