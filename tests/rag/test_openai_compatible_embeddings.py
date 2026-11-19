"""Offline tests for OpenAICompatibleEmbeddingModel (fake client, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_reliability.rag.embeddings import EmbeddingBackendError, OpenAICompatibleEmbeddingModel


class _FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: list[str]):  # noqa: A002 - mirrors the SDK
        self.calls.append(list(input))
        if self.fail:
            raise RuntimeError("server down")
        # Return out of order to check the model re-sorts by index.
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_prefixes_batching_and_order() -> None:
    client = _FakeClient()
    model = OpenAICompatibleEmbeddingModel(
        "nomic-embed-text",
        client=client,
        batch_size=2,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    docs = model.embed(["a", "bb", "ccc"])
    assert [v[0] for v in docs] == [
        len("search_document: a"),
        len("search_document: bb"),
        len("search_document: ccc"),
    ]
    assert len(client.calls) == 2  # batched 2 + 1
    query = model.embed_one("q")
    assert query[0] == len("search_query: q")
    assert model.get_config()["query_prefix"] == "search_query: "


def test_backend_failure_is_explicit() -> None:
    model = OpenAICompatibleEmbeddingModel("m", client=_FakeClient(fail=True))
    with pytest.raises(EmbeddingBackendError, match="server down"):
        model.embed(["x"])
