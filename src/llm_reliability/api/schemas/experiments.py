"""
Pydantic schemas for experiments and controlled benchmark executions.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

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
    # Optional: reference-free cases are supported by the engine (reference-based
    # evaluators report SKIPPED for them; faithfulness can still be judged).
    reference_answer: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateModelConfigRequest(BaseApiSchema):
    model_id: str = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateExperimentRequest(BaseApiSchema):
    """Either inline ``test_cases`` or a bundled ``dataset_preset``, never both.

    With a preset, ``retrieval_top_k`` makes it a RAG experiment over the
    preset's corpus (the server fills ``model_config.extra_params.rag``);
    without it, a closed-book experiment with the matching answer-format
    template. ``judge_model_id`` makes ``llm_judge`` / ``llm_faithfulness_judge``
    real LLM judges on that model.
    """

    experiment_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(min_length=1)
    description: str | None = None
    dataset_name: str | None = Field(default=None, min_length=1)
    test_cases: list[CreateTestCaseRequest] | None = Field(default=None, min_length=1)
    dataset_preset: str | None = None
    dataset_limit: int | None = Field(default=None, ge=1)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=20)
    embedding_model_id: str | None = Field(default=None, min_length=1)
    judge_model_id: str | None = Field(default=None, min_length=1)
    model_config_req: CreateModelConfigRequest = Field(alias="model_config")
    evaluator_names: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_dataset_source(self) -> CreateExperimentRequest:
        if (self.test_cases is None) == (self.dataset_preset is None):
            raise ValueError("Provide exactly one of 'test_cases' or 'dataset_preset'.")
        if self.test_cases is not None and not self.dataset_name:
            raise ValueError("'dataset_name' is required with inline 'test_cases'.")
        if self.dataset_preset is None and (
            self.retrieval_top_k is not None
            or self.dataset_limit is not None
            or self.embedding_model_id is not None
        ):
            raise ValueError(
                "'retrieval_top_k', 'embedding_model_id' and 'dataset_limit' apply only to a "
                "'dataset_preset' (retrieval needs the preset's corpus and labels)."
            )
        return self


class DatasetPresetResponse(BaseApiSchema):
    preset_id: str
    name: str
    description: str
    test_case_count: int
    document_count: int
    has_retrieval_labels: bool
    license: str | None = None
    citation: str | None = None
    corpus_fingerprint: str


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


class JobResponse(BaseApiSchema):
    """State of a background execution. ``completed``/``total`` count generations."""

    job_id: str
    experiment_id: str
    adapter_id: str
    status: str
    total: int
    completed: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    error: str | None = None
