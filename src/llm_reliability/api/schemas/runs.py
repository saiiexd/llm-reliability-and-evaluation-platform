"""
Pydantic schemas for experiment runs and test case execution details.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema
from llm_reliability.api.schemas.experiments import (
    DatasetVersionResponse,
    EvaluatorConfigResponse,
    ModelConfigResponse,
)


class EvaluationResultResponse(BaseApiSchema):
    evaluator_name: str
    criterion: str
    passed: bool | None = None
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    status: str
    error: str | None = None
    duration_seconds: float | None = None


class TestCaseResultResponse(BaseApiSchema):
    __test__ = False

    test_case_id: str
    input_text: str
    reference_answer: str | None = None
    output_text: str | None = None
    passed: bool
    succeeded: bool = True
    evaluations: list[EvaluationResultResponse] = Field(default_factory=list)
    evaluation_results: list[EvaluationResultResponse] = Field(default_factory=list)
    duration_seconds: float | None = None
    execution_error: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummaryResponse(BaseApiSchema):
    run_id: str
    experiment_id: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float | None = None
    config_fingerprint: str
    num_test_cases: int
    num_succeeded: int
    num_failed: int
    num_passed: int | None = None
    pass_rate: float | None = None
    error: str | None = None
    adapter_id: str | None = None
    executed_model_ids: list[str] = Field(default_factory=list)
    simulated_components: list[str] = Field(default_factory=list)


class RunDetailResponse(BaseApiSchema):
    run_id: str
    experiment_id: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float | None = None
    config_fingerprint: str
    dataset_version: DatasetVersionResponse
    model_config_detail: ModelConfigResponse = Field(alias="model_config")
    evaluator_configs: list[EvaluatorConfigResponse]
    num_test_cases: int
    num_succeeded: int
    num_failed: int
    num_passed: int | None = None
    pass_rate: float | None = None
    test_case_results: list[TestCaseResultResponse] = Field(default_factory=list)
    error: str | None = None
    adapter_id: str | None = None
    executed_model_ids: list[str] = Field(default_factory=list)
    simulated_components: list[str] = Field(default_factory=list)


class RunSummariesResponse(BaseApiSchema):
    run_id: str
    evaluation_summary: dict[str, Any]
    reliability_findings: list[dict[str, Any]]
    retrieval_summary: dict[str, Any] = Field(default_factory=dict)
