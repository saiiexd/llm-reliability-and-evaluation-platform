"""
Health readiness and system status endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from llm_reliability.api.config import ApiSettings
from llm_reliability.api.dependencies import get_repository, get_settings
from llm_reliability.api.schemas.common import HealthResponse
from llm_reliability.experiments.repository import ExperimentRepository

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health(
    settings: ApiSettings = Depends(get_settings),
    repository: ExperimentRepository = Depends(get_repository),
) -> HealthResponse:
    """Confirms application readiness and basic repository stats without leaking secrets."""
    total_experiments = len(repository.list_experiment_ids())
    total_runs = len(repository.list_run_ids())

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(UTC).isoformat(),
        total_experiments=total_experiments,
        total_runs=total_runs,
        environment=settings.environment,
    )
