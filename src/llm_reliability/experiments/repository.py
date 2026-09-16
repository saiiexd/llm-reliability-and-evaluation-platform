"""
Persistence interface for experiments and runs.

``ExperimentRepository`` is the only boundary between the Experiment System
and however experiment definitions and runs are actually stored. Domain
logic (``ExperimentManager`` and the models it works with) depends only on
this interface, never on filesystem or database specifics, so
``LocalFileExperimentRepository`` (this milestone) can later be replaced by
a database-backed implementation without changing anything above this
boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_reliability.experiments.models import ExperimentDefinition, ExperimentRun


class ExperimentRepositoryError(Exception):
    """Base class for all experiment persistence errors."""


class ExperimentNotFoundError(ExperimentRepositoryError):
    """Raised when a requested experiment id does not exist in the store."""


class RunNotFoundError(ExperimentRepositoryError):
    """Raised when a requested run id does not exist in the store."""


class ExperimentAlreadyExistsError(ExperimentRepositoryError):
    """Raised when saving an experiment would overwrite an existing, different id."""


class RunAlreadyExistsError(ExperimentRepositoryError):
    """Raised when saving a run would overwrite an existing, different id."""


class CorruptedRecordError(ExperimentRepositoryError):
    """Raised when a persisted record cannot be parsed back into a valid domain model."""


class ExperimentRepository(ABC):
    """Storage interface for experiment definitions and their runs."""

    @abstractmethod
    def save_experiment(self, definition: ExperimentDefinition) -> None:
        """Persist ``definition``.

        Raises ``ExperimentAlreadyExistsError`` if ``definition.experiment_id``
        is already stored; an experiment definition is never silently
        overwritten.
        """
        raise NotImplementedError

    @abstractmethod
    def load_experiment(self, experiment_id: str) -> ExperimentDefinition:
        """Load and validate the experiment definition stored as ``experiment_id``.

        Raises ``ExperimentNotFoundError`` if it does not exist, or
        ``CorruptedRecordError`` if the stored record cannot be validated
        back into an ``ExperimentDefinition``.
        """
        raise NotImplementedError

    @abstractmethod
    def save_run(self, run: ExperimentRun) -> None:
        """Persist ``run``.

        Raises ``RunAlreadyExistsError`` if ``run.run_id`` is already
        stored; a run record is never silently overwritten.
        """
        raise NotImplementedError

    @abstractmethod
    def load_run(self, run_id: str) -> ExperimentRun:
        """Load and validate the run stored as ``run_id``.

        Raises ``RunNotFoundError`` if it does not exist, or
        ``CorruptedRecordError`` if the stored record cannot be validated
        back into an ``ExperimentRun``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_experiment_ids(self) -> list[str]:
        """Return all stored experiment ids, in no particular order."""
        raise NotImplementedError

    @abstractmethod
    def list_run_ids(self, experiment_id: str | None = None) -> list[str]:
        """Return stored run ids, optionally filtered to one experiment."""
        raise NotImplementedError
