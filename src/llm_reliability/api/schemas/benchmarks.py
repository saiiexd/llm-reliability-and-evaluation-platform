"""
Pydantic schemas for multi-model benchmark executions.
"""

from __future__ import annotations

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema
from llm_reliability.api.schemas.experiments import RunExecutionResponse


class ExecuteBenchmarkRequest(BaseApiSchema):
    """Request to execute a controlled multi-model benchmark with repetitions."""

    experiment_id: str = Field(min_length=1)
    adapter_ids: list[str] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, le=10)
    allow_external_inference: bool = Field(default=False)
    description: str | None = None


class BenchmarkExecutionResponse(BaseApiSchema):
    """Response containing execution results for all benchmark runs."""

    benchmark_id: str
    experiment_id: str
    total_runs: int
    runs: list[RunExecutionResponse]
