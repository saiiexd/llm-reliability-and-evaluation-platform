"""
Pydantic schemas for reliability diagnostics and forensic claim assessments.
"""

from __future__ import annotations

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema


class ClaimAssessmentResponse(BaseApiSchema):
    statement: str
    status: str
    method: str | None = None
    confidence: float | None = None
    evidence_snippets: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class ReliabilityDiagnosticResponse(BaseApiSchema):
    test_case_id: str
    run_id: str
    failure_category: str | None = None
    secondary_findings: list[str] = Field(default_factory=list)
    status: str
    confidence: float | None = None
    explanation: str
    question: str
    reference_answer: str | None = None
    generated_answer: str | None = None
    retrieved_context: list[str] | None = None
    contributing_evaluators: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessmentResponse] = Field(default_factory=list)
    error: str | None = None
