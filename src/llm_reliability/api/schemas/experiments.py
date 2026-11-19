"""
Pydantic schemas for experiments and controlled benchmark executions.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema


class DatasetVersionResponse(BaseApiSchema):
    version_id: str
    dataset_name: str
    test_case_count: int


class ModelConfigResponse(BaseApiSchema):
    model_id: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfigResponse(BaseApiSchema):
    evaluator_name: str
    criterion: str = ""
    evaluator_version: str = "1.0.0"
    config: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentSummaryResponse(BaseApiSchema):
    experiment_id: str
    name: str
    description: str | None = None
    created_at: str
    config_fingerprint: str
    dataset_name: str
    test_case_count: int
    model_id: str
    evaluator_count: int
    runs_count: int


class ExperimentDetailResponse(BaseApiSchema):
    experiment_id: str
    name: str
    description: str | None = None
    created_at: str
    config_fingerprint: str
    dataset_version: DatasetVersionResponse
    model_config_detail: ModelConfigResponse = Field(alias="model_config")
    evaluator_configs: list[EvaluatorConfigResponse]
    metadata: dict[str, Any] = Field(default_factory=dict)
    runs_count: int = 0


class CreateTestCaseRequest(BaseApiSchema):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateModelConfigRequest(BaseApiSchema):
    model_id: str = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateExperimentRequest(BaseApiSchema):
    experiment_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(min_length=1)
    description: str | None = None
    dataset_name: str = Field(min_length=1)
    test_cases: list[CreateTestCaseRequest] = Field(min_length=1)
    model_config_req: CreateModelConfigRequest = Field(alias="model_config")
    evaluator_names: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecuteExperimentRequest(BaseApiSchema):
    """Execution request referencing an allowlisted adapter ID."""

    adapter_id: str = Field(
        default="mock-adapter-v1",
        description="Allowlisted adapter identifier from the adapter registry.",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional custom run ID (auto-generated if omitted).",
    )
    allow_external_inference: bool = Field(
        default=False,
        description="Explicit opt-in to allow outbound external model inference API calls.",
    )


class RunExecutionResponse(BaseApiSchema):
    run_id: str
    experiment_id: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float | None = None
    num_succeeded: int
    num_failed: int
    num_passed: int | None = None
    pass_rate: float | None = None
    error: str | None = None
