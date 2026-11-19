"""
Common schemas, pagination envelopes, and redaction helpers.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SECRET_PATTERN = re.compile(r"(key|secret|token|password|auth|credential|bearer)", re.IGNORECASE)


# Stricter than _SECRET_PATTERN (which is deliberately broad for *display*
# redaction and would also match harmless keys such as "max_tokens"): these are
# key names that can only plausibly hold a credential, so a request containing
# them is rejected instead of being written to the experiment store.
_CREDENTIAL_KEY_PATTERN = re.compile(
    r"(^|[_-])(api[_-]?key|secret|password|passwd|credentials?|bearer|authorization|"
    r"access[_-]?token|auth[_-]?token|token)$",
    re.IGNORECASE,
)


def find_credential_keys(data: Any, path: str = "") -> list[str]:
    """Return dotted paths of keys in ``data`` that look like credentials."""
    found: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            key_path = f"{path}.{k}" if path else str(k)
            if _CREDENTIAL_KEY_PATTERN.search(str(k)):
                found.append(key_path)
            found.extend(find_credential_keys(v, key_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            found.extend(find_credential_keys(item, f"{path}[{i}]"))
    return found


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
