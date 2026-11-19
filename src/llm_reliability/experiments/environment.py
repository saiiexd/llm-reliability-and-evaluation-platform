"""
Execution environment capture for run provenance.

A configuration fingerprint identifies *what* was asked for; it says nothing
about the software that carried it out. Two runs with the same fingerprint
can differ because the platform code, a dependency, or the interpreter
changed between them. ``capture_environment`` records those facts at
execution time so a reader can tell "same configuration, same software" from
"same configuration, different software".

Deliberately *not* part of the configuration fingerprint: the environment is
an execution condition, like the adapter id, not part of the experiment's
definition.

Best effort by design: every field degrades to ``None`` instead of failing
execution (for example, ``git_commit`` is ``None`` when the package is
installed outside a Git checkout). It never records environment variables,
paths under the user's home, host names or anything credential-like.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGE_NAME = "llm-reliability-platform"
# Libraries whose version can change evaluation or provider behaviour.
TRACKED_DEPENDENCIES = (
    "fastapi",
    "pydantic",
    "openai",
    "numpy",
    "bert-score",
    "sentence-transformers",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def capture_environment() -> dict[str, Any]:
    """Return a JSON-serializable record of the software that executes a run."""
    commit = _git("rev-parse", "HEAD")
    dirty: bool | None = None
    if commit:
        status = _git("status", "--porcelain", "--", "src")
        dirty = bool(status) if status is not None else None
    dependencies = {name: _version(name) for name in TRACKED_DEPENDENCIES}
    return {
        "platform_version": _version(PACKAGE_NAME),
        "git_commit": commit,
        "git_dirty_src": dirty,
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "os": platform.system(),
        "machine": platform.machine(),
        "dependencies": {k: v for k, v in dependencies.items() if v is not None},
    }
