"""
The bundled SQuAD v1.1 development-set subset (``data/benchmarks``).

One loader shared by the EXP-001 study script and the API, so a dataset
built from the console is byte-identical to the one the study used (same
test cases, same derived retrieval labels, hence the same dataset version
fingerprint).

Retrieval labels (``relevant_chunk_ids``) are *derived*, not annotated: a
chunk is relevant when it fully contains the first gold answer span (falling
back to chunks that overlap it). They therefore depend on the chunking, which
is why ``dataset(chunking)`` takes the chunking as an argument and records the
rule in each test case's metadata.

License: SQuAD is CC BY-SA 4.0 (Rajpurkar et al., 2016).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_reliability.evaluation.models import Dataset, TestCase
from llm_reliability.rag.documents import Chunk, ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.pipeline import compute_corpus_fingerprint

SQUAD_SUBSET_ID = "squad-v1.1-dev-subset-v1"
SQUAD_SUBSET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "squad_v1_1_dev_subset_v1.json"
)
# The chunking EXP-001 used; relevant_chunk_ids below are derived for it.
SQUAD_CHUNKING = ChunkingConfig(chunk_size=800, chunk_overlap=100)
RELEVANCE_RULE = "chunks fully containing the first gold answer span"


def _chunk_span(chunk: Chunk, chunking: ChunkingConfig) -> tuple[int, int]:
    step = chunking.chunk_size - chunking.chunk_overlap
    start = chunk.position * step
    return start, start + len(chunk.text)


@dataclass(frozen=True)
class SquadSubset:
    """The raw subset file plus helpers to build its corpus and dataset."""

    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def description(self) -> str:
        return str(self.raw.get("description", ""))

    @property
    def source(self) -> dict[str, Any]:
        return dict(self.raw.get("source", {}))

    def documents(self) -> list[Document]:
        return [Document(id=d["id"], content=d["content"]) for d in self.raw["documents"]]

    def corpus_fingerprint(self) -> str:
        return compute_corpus_fingerprint(self.documents())

    def chunks(self, chunking: ChunkingConfig = SQUAD_CHUNKING) -> list[Chunk]:
        return FixedSizeChunker(chunking).chunk_documents(self.documents())

    def dataset(
        self, chunking: ChunkingConfig = SQUAD_CHUNKING, limit: int | None = None
    ) -> Dataset:
        """Build the dataset with relevant_chunk_ids derived for ``chunking``."""
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.chunks(chunking):
            by_doc.setdefault(chunk.document_id, []).append(chunk)
        cases = self.raw["test_cases"][:limit] if limit else self.raw["test_cases"]
        test_cases = []
        for tc in cases:
            meta = dict(tc["metadata"])
            a_start = meta["answer_start"]
            a_end = a_start + len(tc["reference_answer"])
            doc_chunks = by_doc[meta["source_document_id"]]
            spans = {c.id: _chunk_span(c, chunking) for c in doc_chunks}
            containing = [
                c.id for c in doc_chunks if spans[c.id][0] <= a_start <= a_end <= spans[c.id][1]
            ]
            overlapping = [
                c.id for c in doc_chunks if spans[c.id][0] < a_end and a_start < spans[c.id][1]
            ]
            meta["relevant_chunk_ids"] = containing or overlapping
            meta["relevance_rule"] = RELEVANCE_RULE
            test_cases.append(
                TestCase(
                    id=tc["id"],
                    input=tc["input"],
                    reference_answer=tc["reference_answer"],
                    metadata=meta,
                )
            )
        suffix = f"-first{limit}" if limit else ""
        return Dataset(name=f"{self.name}{suffix}", test_cases=test_cases)


@lru_cache(maxsize=1)
def load_squad_subset(path: Path = SQUAD_SUBSET_PATH) -> SquadSubset:
    return SquadSubset(raw=json.loads(path.read_text(encoding="utf-8")))
