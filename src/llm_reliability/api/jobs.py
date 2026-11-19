"""
In-process background execution for long runs.

A 60-question run against a real model takes minutes; holding an HTTP request
open that long blocks the console. A job runs the same
``ExperimentManager.execute_experiment`` call on a worker thread and exposes
progress (questions generated so far) for polling.

Deliberately minimal, per the project's "no unnecessary infrastructure" rule:
one worker thread (runs execute one at a time, so they do not compete for a
local GPU and distort each other's latency), job state in memory. Jobs are
therefore lost when the backend restarts; a run is only persisted when it
finishes, exactly as with synchronous execution, so a restart can lose an
in-flight run but can never leave a partial run on disk.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse
from llm_reliability.experiments.models import ExperimentRun


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ProgressAdapter(ModelAdapter):
    """Counts generations; forwards everything else (``config``, ``template_id``)."""

    def __init__(self, inner: ModelAdapter, on_progress: Callable[[], None]) -> None:
        self._inner = inner
        self._on_progress = on_progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        try:
            return self._inner.generate(request)
        finally:
            self._on_progress()


@dataclass
class Job:
    job_id: str
    experiment_id: str
    adapter_id: str
    total: int
    status: str = "queued"  # queued | running | succeeded | failed
    completed: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class JobManager:
    """Runs execution callables on one worker thread and tracks their state."""

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        experiment_id: str,
        adapter_id: str,
        total: int,
        execute: Callable[[Callable[[], None]], ExperimentRun],
    ) -> Job:
        """Queue ``execute(progress_callback)``; it must return the persisted run."""
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            experiment_id=experiment_id,
            adapter_id=adapter_id,
            total=total,
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def progress() -> None:
            with self._lock:
                job.completed = min(job.total, job.completed + 1)

        def work() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = _now()
            try:
                run = execute(progress)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_at = _now()
                return
            with self._lock:
                job.status = "succeeded"
                job.run_id = run.run_id
                job.run_status = run.status.value
                job.error = run.error
                job.finished_at = _now()

        self._executor.submit(work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def snapshot(self, job: Job) -> dict[str, Any]:
        with self._lock:
            return job.to_dict()
