"""Versioned benchmark datasets bundled with the repository (see ``data/benchmarks``)."""

from llm_reliability.datasets.squad import (
    SQUAD_CHUNKING,
    SQUAD_SUBSET_ID,
    SquadSubset,
    load_squad_subset,
)

__all__ = ["SQUAD_CHUNKING", "SQUAD_SUBSET_ID", "SquadSubset", "load_squad_subset"]
