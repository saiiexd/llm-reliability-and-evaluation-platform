"""
Background execution status endpoints (see ``llm_reliability.api.jobs``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_reliability.api.dependencies import get_job_manager
from llm_reliability.api.errors import ResourceNotFoundError
from llm_reliability.api.jobs import JobManager
from llm_reliability.api.schemas.experiments import JobResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs(jobs: JobManager = Depends(get_job_manager)) -> list[JobResponse]:
    """Jobs started since the backend started, newest first (state is in memory)."""
    return [JobResponse(**jobs.snapshot(job)) for job in jobs.list()]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, jobs: JobManager = Depends(get_job_manager)) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise ResourceNotFoundError(resource_type="Job", resource_id=job_id)
    return JobResponse(**jobs.snapshot(job))
