"""
RAG execution and retrieval evaluation.

Implements the explicit execution model::

    Question -> Retrieval -> Retrieved Context -> Generation -> Answer

as a controlled research and engineering pipeline, not a generic RAG
framework or chatbot. Retrieval and generation are separate, inspectable
stages (``Retriever`` and the existing ``ModelAdapter``, composed by
``RagPipeline``), and retrieval quality is evaluated independently from
answer quality (``RetrievalEvaluator`` / ``RetrievalEvaluationResult`` vs.
the existing ``Evaluator`` / ``EvaluationResult``).

``RagPipeline`` implements the existing ``ModelAdapter`` interface, so it
runs through the unmodified ``EvaluationRunner`` and ``ExperimentManager``
from earlier milestones with no changes to either. Retrieval evidence
travels through the existing ``ModelResponse.provider_metadata`` extension
point; retrieval ground truth travels through the existing
``TestCase.metadata``. No persisted schema changed to support this
milestone.

See ``research/notes/rag_execution_model.md`` for the full conceptual
model, including why retrieval and generation failure must be diagnosed
separately and what ``diagnose_rag_execution`` does and does not claim to
determine.
"""

from llm_reliability.rag.context_support import ContextSupportBaseline, compute_overlap_ratio
from llm_reliability.rag.diagnosis import RagDiagnosis, RagDiagnosisCategory, diagnose_rag_execution
from llm_reliability.rag.documents import Chunk, ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.embeddings import (
    EmbeddingBackendError,
    EmbeddingModel,
    FakeEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from llm_reliability.rag.evaluators import (
    HitAtKEvaluator,
    MeanReciprocalRankEvaluator,
    RecallAtKEvaluator,
    RetrievalEvaluationResult,
    RetrievalEvaluationStatus,
    RetrievalEvaluator,
    evaluate_retrieval,
)
from llm_reliability.rag.faithfulness_judge import (
    FAITHFULNESS_RUBRIC,
    LLMFaithfulnessEvaluator,
    render_faithfulness_prompt,
)
from llm_reliability.rag.ground_truth import RELEVANT_CHUNK_IDS_KEY, get_relevant_chunk_ids
from llm_reliability.rag.pipeline import (
    RagPipeline,
    RagPipelineConfig,
    build_rag_prompt,
    compute_corpus_fingerprint,
    extract_rag_metadata,
    extract_retrieved_chunks,
)
from llm_reliability.rag.retriever import (
    RetrievalConfig,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
)
from llm_reliability.rag.summary import RetrievalEvaluatorSummary, summarize_retrieval

__all__ = [
    "FAITHFULNESS_RUBRIC",
    "RELEVANT_CHUNK_IDS_KEY",
    "Chunk",
    "ChunkingConfig",
    "ContextSupportBaseline",
    "Document",
    "EmbeddingBackendError",
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "FixedSizeChunker",
    "HitAtKEvaluator",
    "LLMFaithfulnessEvaluator",
    "MeanReciprocalRankEvaluator",
    "RagDiagnosis",
    "RagDiagnosisCategory",
    "RagPipeline",
    "RagPipelineConfig",
    "RecallAtKEvaluator",
    "RetrievalConfig",
    "RetrievalEvaluationResult",
    "RetrievalEvaluationStatus",
    "RetrievalEvaluator",
    "RetrievalEvaluatorSummary",
    "RetrievalResult",
    "RetrievedChunk",
    "Retriever",
    "SentenceTransformerEmbeddingModel",
    "build_rag_prompt",
    "compute_corpus_fingerprint",
    "compute_overlap_ratio",
    "diagnose_rag_execution",
    "evaluate_retrieval",
    "extract_rag_metadata",
    "extract_retrieved_chunks",
    "get_relevant_chunk_ids",
    "render_faithfulness_prompt",
    "summarize_retrieval",
]
