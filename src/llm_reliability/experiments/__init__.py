"""
Experiment System.

Builds a reproducible experiment on top of the Core Evaluation Engine
(``llm_reliability.evaluation``):

    ExperimentDefinition (DatasetVersion + ModelConfig + EvaluatorConfig(s))
        -> ExperimentManager.execute_experiment (delegates to EvaluationRunner)
        -> ExperimentRun (status + full RunResult snapshot)
        -> ExperimentRepository (local JSON persistence in this milestone)

See ``research/notes/`` for the conceptual model and reproducibility
strategy. This package does not implement a database, a dashboard, RAG
evaluation, or reliability/statistical analysis; those are later
milestones.
"""

from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)
from llm_reliability.experiments.repository import (
    CorruptedRecordError,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRepository,
    ExperimentRepositoryError,
    RunAlreadyExistsError,
    RunNotFoundError,
)

__all__ = [
    "CorruptedRecordError",
    "DatasetVersion",
    "EvaluatorConfig",
    "ExperimentAlreadyExistsError",
    "ExperimentDefinition",
    "ExperimentManager",
    "ExperimentNotFoundError",
    "ExperimentRepository",
    "ExperimentRepositoryError",
    "ExperimentRun",
    "LocalFileExperimentRepository",
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "RunStatus",
]
