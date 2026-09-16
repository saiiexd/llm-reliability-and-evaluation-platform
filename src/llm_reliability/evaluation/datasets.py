"""
Loading ``Dataset`` objects from JSON files on disk.

Small, reusable complement to ``Dataset.from_dict``: reads a JSON file
matching that shape (``{"name": ..., "test_cases": [...]}}``) and returns a
validated ``Dataset``. Used to load version-controlled, curated dataset
files such as ``data/eval_sets/evaluation_methodology_baseline_v1.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_reliability.evaluation.models import Dataset


def load_dataset_from_json(path: str | Path) -> Dataset:
    """Load and validate a ``Dataset`` from a JSON file.

    Raises the same explicit errors as ``Dataset.from_dict``/``TestCase``
    construction if the file's content is malformed.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dataset.from_dict(data)
