"""Fast unit tests for the embedding plumbing — the factory and the
completeness guard. These use a deterministic fake embedder so they need no
model download and run in CI on the `dev` extra alone. The real MiniLM backend
is exercised separately in test_embeddings_local.py.
"""

import pytest

from edgar_rag.config import Settings
from edgar_rag.embeddings import embed_chunks, get_embedding_provider
from edgar_rag.embeddings.local import LocalEmbeddingProvider
from edgar_rag.models import Chunk


class FakeEmbedder:
    """Stands in for a real embedder without the heavy model download.

    Deterministic: each vector encodes the text length so tests can assert on
    exact values, and there is always exactly one vector per input text.
    """

    name = "fake"
    dim = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            document_id="doc",
            tenant="acme",
            doc_type="10-K",
            page_number=1,
            chunk_index=i,
            text=f"section {i} body text",
        )
        for i in range(n)
    ]


def test_embed_chunks_fills_every_chunk():
    chunks = _chunks(5)
    embed_chunks(FakeEmbedder(), chunks)
    assert all(chunk.embedding is not None and len(chunk.embedding) == 3 for chunk in chunks)


def test_embed_chunks_rejects_a_provider_that_drops_vectors():
    class DroppingProvider:
        name = "dropping"
        dim = 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Simulates the production bug: fewer vectors than inputs.
            return [[0.0, 0.0, 0.0] for _ in texts[:-1]]

    with pytest.raises(ValueError, match="count mismatch"):
        embed_chunks(DroppingProvider(), _chunks(3))


def test_embed_chunks_on_empty_list_is_a_noop():
    assert embed_chunks(FakeEmbedder(), []) == []


def test_factory_builds_local_without_downloading_a_model():
    # LocalEmbeddingProvider defers the model load to the first embed call, so
    # the factory can be exercised in CI without pulling ~90 MB.
    provider = get_embedding_provider(Settings(embedding_provider="local"))
    assert isinstance(provider, LocalEmbeddingProvider)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown embedding_provider"):
        get_embedding_provider(Settings(embedding_provider="nope"))
