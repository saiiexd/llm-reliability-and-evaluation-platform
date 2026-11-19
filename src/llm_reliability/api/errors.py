"""
API exception hierarchy and error handling definitions.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Predictable error envelope for API error responses."""

    code: str = Field(description="Machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation.")
    details: Any = Field(default=None, description="Optional structured metadata or field errors.")


class ErrorResponse(BaseModel):
    """Top-level error response model."""

    error: ErrorDetail


class ApiException(HTTPException):
    """Base API exception that maps directly to the standard error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message, "details": details},
            headers=headers,
        )
        self.code = code
        self.message = message
        self.details = details


class ResourceNotFoundError(ApiException):
    """Raised when an experiment, run, or test case does not exist."""

    def __init__(self, resource_type: str, resource_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="RESOURCE_NOT_FOUND",
            message=f"{resource_type} '{resource_id}' was not found.",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ConflictError(ApiException):
    """Raised when an immutable experiment or run ID already exists."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            code="RESOURCE_CONFLICT",
            message=message,
            details=details,
        )


class ExecutionError(ApiException):
    """Raised when a benchmark or evaluation execution fails."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="EXECUTION_ERROR",
            message=message,
            details=details,
        )


class ConfigurationSecurityError(ApiException):
    """Raised when an untrusted or non-allowlisted configuration is submitted."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="CONFIGURATION_SECURITY_ERROR",
            message=message,
            details=details,
        )
