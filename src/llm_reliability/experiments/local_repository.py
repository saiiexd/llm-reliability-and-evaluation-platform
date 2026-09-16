"""
Local filesystem implementation of ``ExperimentRepository``.

Stores each experiment definition and each run as one human-readable JSON
file. This exists to validate the experiment domain model and
reproducibility workflow before a production database is introduced; it is
deliberately simple (no indexing, no query language, no concurrency
control beyond what the filesystem itself provides) and is not intended to
scale beyond local development and small-scale research use.

Layout, rooted at a caller-supplied directory::

    <root>/
      experiments/
        <experiment_id>.json
      runs/
        <experiment_id>/
          <run_id>.json

Writes are atomic per file: content is written to a temporary file in the
same directory and then moved into place with ``os.replace``, which is
atomic on both POSIX and Windows for a source and destination on the same
filesystem. This means a process interrupted mid-write can never leave a
half-written JSON file at the target path; either the old content (if any)
remains, or the new content is fully present.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from llm_reliability.experiments.models import ExperimentDefinition, ExperimentRun
from llm_reliability.experiments.repository import (
    CorruptedRecordError,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRepository,
    ExperimentRepositoryError,
    RunAlreadyExistsError,
    RunNotFoundError,
)

_FORBIDDEN_IDENTIFIER_CHARS = ("/", "\\")


def _validate_identifier(identifier: str, label: str) -> None:
    if not identifier or not identifier.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    if identifier in (".", ".."):
        raise ValueError(f"{label} {identifier!r} is not a valid identifier.")
    if any(char in identifier for char in _FORBIDDEN_IDENTIFIER_CHARS):
        raise ValueError(
            f"{label} {identifier!r} must not contain path separators; "
            "it is used directly as a filename."
        )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


class LocalFileExperimentRepository(ExperimentRepository):
    """Filesystem-backed ``ExperimentRepository`` storing one JSON file per record."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._experiments_dir = self._root / "experiments"
        self._runs_dir = self._root / "runs"

    def _experiment_path(self, experiment_id: str) -> Path:
        _validate_identifier(experiment_id, "experiment_id")
        return self._experiments_dir / f"{experiment_id}.json"

    def _run_path(self, experiment_id: str, run_id: str) -> Path:
        _validate_identifier(experiment_id, "experiment_id")
        _validate_identifier(run_id, "run_id")
        return self._runs_dir / experiment_id / f"{run_id}.json"

    def save_experiment(self, definition: ExperimentDefinition) -> None:
        path = self._experiment_path(definition.experiment_id)
        if path.exists():
            raise ExperimentAlreadyExistsError(
                f"Experiment '{definition.experiment_id}' already exists at {path}. "
                "Experiment definitions are immutable once saved; use a different "
                "experiment_id for a changed definition."
            )
        _atomic_write_json(path, definition.to_dict())

    def load_experiment(self, experiment_id: str) -> ExperimentDefinition:
        path = self._experiment_path(experiment_id)
        if not path.exists():
            raise ExperimentNotFoundError(f"No experiment found with id '{experiment_id}'.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ExperimentDefinition.from_dict(data)
        except ExperimentRepositoryError:
            raise
        except Exception as exc:
            raise CorruptedRecordError(
                f"Experiment record at {path} could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc

    def save_run(self, run: ExperimentRun) -> None:
        path = self._run_path(run.experiment_id, run.run_id)
        if path.exists():
            raise RunAlreadyExistsError(
                f"Run '{run.run_id}' for experiment '{run.experiment_id}' already exists "
                f"at {path}. Run records are immutable once saved."
            )
        _atomic_write_json(path, run.to_dict())

    def load_run(self, run_id: str) -> ExperimentRun:
        path = self._find_run_path(run_id)
        if path is None:
            raise RunNotFoundError(f"No run found with id '{run_id}'.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ExperimentRun.from_dict(data)
        except ExperimentRepositoryError:
            raise
        except Exception as exc:
            raise CorruptedRecordError(
                f"Run record at {path} could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc

    def _find_run_path(self, run_id: str) -> Path | None:
        _validate_identifier(run_id, "run_id")
        if not self._runs_dir.is_dir():
            return None
        for experiment_dir in self._runs_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            candidate = experiment_dir / f"{run_id}.json"
            if candidate.exists():
                return candidate
        return None

    def list_experiment_ids(self) -> list[str]:
        if not self._experiments_dir.is_dir():
            return []
        return sorted(p.stem for p in self._experiments_dir.glob("*.json"))

    def list_run_ids(self, experiment_id: str | None = None) -> list[str]:
        if not self._runs_dir.is_dir():
            return []
        if experiment_id is not None:
            _validate_identifier(experiment_id, "experiment_id")
            experiment_run_dir = self._runs_dir / experiment_id
            if not experiment_run_dir.is_dir():
                return []
            return sorted(p.stem for p in experiment_run_dir.glob("*.json"))
        run_ids: list[str] = []
        for experiment_dir in self._runs_dir.iterdir():
            if experiment_dir.is_dir():
                run_ids.extend(p.stem for p in experiment_dir.glob("*.json"))
        return sorted(run_ids)
