"""
Pydantic schemas for robustness perturbations and consistency analyses.
"""

from __future__ import annotations

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema


class RobustnessComparisonResponse(BaseApiSchema):
    baseline_test_case_id: str
    perturbed_test_case_id: str
    perturbation_type: str
    baseline_category: str | None = None
    perturbed_category: str | None = None
    outcome_changed: bool
    category_changed: bool
    failure_category_transition: str | None = None


class RobustnessConsistencyResponse(BaseApiSchema):
    total_comparisons: int
    identical_outcomes: int
    consistency_rate: float | None = None
    category_transitions: dict[str, int] = Field(default_factory=dict)
    transitions_by_type: dict[str, dict[str, int]] = Field(default_factory=dict)


class RobustnessBundleResponse(BaseApiSchema):
    run_id: str
    comparisons: list[RobustnessComparisonResponse] = Field(default_factory=list)
    consistency: RobustnessConsistencyResponse | None = None
