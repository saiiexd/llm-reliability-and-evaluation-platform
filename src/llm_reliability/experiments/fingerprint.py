"""
Deterministic content fingerprinting for reproducibility records.

This module is intentionally free of any dependency on the domain models in
``experiments.models``: callers pass plain, already-serializable Python data
(dicts, lists, primitives), and this module turns that data into a stable
SHA-256 hex digest. Keeping it dependency-free avoids a circular import
between this module and ``models.py``, and keeps the hashing strategy
independently testable.

Strategy: canonical JSON (sorted keys, no incidental whitespace) hashed with
SHA-256. This is not a Merkle tree or a version-control system; it is the
smallest mechanism that satisfies the actual requirement here, which is that
identical content always produces the identical fingerprint, and that
fingerprints are computed only from meaningful content, never from
dictionary/serialization ordering or from runtime-specific fields such as
timestamps or run identifiers (callers are responsible for excluding those
before calling this module).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` as canonical JSON.

    Keys are sorted and separators are minimal, so two payloads that are
    equal as Python data always produce byte-identical JSON regardless of
    dict insertion order.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_fingerprint(payload: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_dataset_fingerprint(test_cases_payload: list[dict[str, Any]]) -> str:
    """Compute a content fingerprint for a dataset's test cases.

    Deliberately excludes the dataset's human-readable name: a dataset
    version's identity is the content of its test cases, not the name it is
    currently known by, so renaming a dataset does not spuriously produce a
    new version and a content-identical dataset under a different name is
    recognized as the same version.
    """
    return content_fingerprint({"test_cases": test_cases_payload})


def compute_experiment_fingerprint(
    dataset_version_id: str,
    model_config_payload: dict[str, Any],
    evaluator_configs_payload: list[dict[str, Any]],
) -> str:
    """Compute a content fingerprint for an experiment's reproducibility-relevant configuration.

    Includes only the dataset version id (already a content fingerprint),
    the model/application configuration, and the evaluator configurations.
    Deliberately excludes the experiment id, name, description, free-form
    metadata, and creation timestamp, none of which affect what actually
    gets executed.
    """
    payload = {
        "dataset_version_id": dataset_version_id,
        "model_config": model_config_payload,
        "evaluator_configs": evaluator_configs_payload,
    }
    return content_fingerprint(payload)
