"""
FastAPI application boundary for the LLM Reliability & Evaluation Platform.

Exposes versioned REST endpoints (/api/v1) for inspecting experiments,
retrieving runs, querying diagnostics, evaluating RAG traces, running
cross-experiment analytics, and performing controlled benchmark executions.
"""

from llm_reliability.api.app import create_app

__all__ = ["create_app"]
