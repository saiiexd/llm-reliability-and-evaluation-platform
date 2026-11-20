"""
FastAPI dependency injection providers.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from llm_reliability.analytics.service import AnalyticsService
from llm_reliability.api.config import ApiSettings
from llm_reliability.api.jobs import JobManager
from llm_reliability.api.registry import AdapterRegistry, DatasetRegistry, EvaluatorRegistry
from llm_reliability.api.services import DiagnosticsService
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.repository import ExperimentRepository


@lru_cache
def get_settings() -> ApiSettings:
    """Cached singleton application settings."""
    return ApiSettings()


@lru_cache
def get_adapter_registry() -> AdapterRegistry:
    """Cached singleton allowlisted adapter registry."""
    return AdapterRegistry()


@lru_cache
def get_evaluator_registry() -> EvaluatorRegistry:
    """Cached singleton allowlisted evaluator registry."""
    return EvaluatorRegistry()


@lru_cache
def get_dataset_registry() -> DatasetRegistry:
    """Cached singleton bundled-dataset registry."""
    return DatasetRegistry()


@lru_cache
def get_job_manager() -> JobManager:
    """Cached singleton background job runner (one worker thread)."""
    return JobManager()


@lru_cache
def get_diagnostics_service() -> DiagnosticsService:
    """Cached singleton diagnostics and forensic derivation service."""
    return DiagnosticsService()


def get_repository(
    settings: ApiSettings = Depends(get_settings),
) -> ExperimentRepository:
    """Provides the configured local filesystem experiment repository."""
    return LocalFileExperimentRepository(root_dir=settings.persistence_dir)


def get_experiment_manager(
    repository: ExperimentRepository = Depends(get_repository),
) -> ExperimentManager:
    """Provides the experiment orchestrator."""
    return ExperimentManager(repository=repository)


def get_analytics_service(
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> AnalyticsService:
    """Provides the cross-experiment analytics service."""

    def load_diags(run_id: str):
        try:
            run = repository.load_run(run_id)
            return list(diag_service.get_diagnostics(run).values())
        except Exception:
            return []

    def load_retrieval(run_id: str):
        try:
            run = repository.load_run(run_id)
            return [
                res
                for res_list in diag_service.get_retrieval_results(run).values()
                for res in res_list
            ]
        except Exception:
            return []

    def load_robustness(run_id: str):
        try:
            run = repository.load_run(run_id)
            return diag_service.get_robustness(run)
        except Exception:
            return []

    return AnalyticsService(
        repository=repository,
        diagnostics_loader=load_diags,
        retrieval_loader=load_retrieval,
        robustness_loader=load_robustness,
    )
