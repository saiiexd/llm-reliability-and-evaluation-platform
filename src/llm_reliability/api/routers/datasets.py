"""
Bundled benchmark datasets that experiments can be created from.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_reliability.api.dependencies import get_dataset_registry
from llm_reliability.api.registry import DatasetRegistry
from llm_reliability.api.schemas.experiments import DatasetPresetResponse

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=list[DatasetPresetResponse])
def list_dataset_presets(
    registry: DatasetRegistry = Depends(get_dataset_registry),
) -> list[DatasetPresetResponse]:
    """Versioned datasets in data/benchmarks, usable as 'dataset_preset' when creating."""
    return [DatasetPresetResponse(**p) for p in registry.list_presets()]
