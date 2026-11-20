"""
Pydantic schemas for RAG execution traces and retrieval evaluation metrics.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema


class RetrievedChunkResponse(BaseApiSchema):
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float | None = None
    rank: int


class RetrievalEvaluationResultResponse(BaseApiSchema):
    evaluator_name: str
    metric_name: str
    score: float | None = None
    k: int | None = None
    status: str
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    evaluator_config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RagTestCaseResultResponse(BaseApiSchema):
    test_case_id: str
    run_id: str
    question: str
    generated_answer: str | None = None
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    retrieval_evaluations: list[RetrievalEvaluationResultResponse] = Field(default_factory=list)
