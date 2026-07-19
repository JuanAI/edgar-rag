import math

import pytest

from edgar_rag.config import Settings
from edgar_rag.embeddings import (
    HashingEmbeddingProvider,
    embed_chunks,
    get_embedding_provider,
)
from edgar_rag.models import Chunk


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_is_deterministic_across_instances():
    text = "Risk factors and market competition."
    first = HashingEmbeddingProvider(dim=64).embed([text])[0]
    second = HashingEmbeddingProvider(dim=64).embed([text])[0]
    assert first == second  # stable hash (blake2b), not process-salted hash()


def test_hashing_vectors_have_configured_dim_and_unit_norm():
    provider = HashingEmbeddingProvider(dim=128)
    vector = provider.embed(["Item 1A. Risk Factors"])[0]
    assert len(vector) == 128
    assert provider.dim == 128
    assert _norm(vector) == pytest.approx(1.0)


def test_hashing_empty_text_is_a_zero_vector_of_correct_dim():
    vector = HashingEmbeddingProvider(dim=32).embed([""])[0]
    assert len(vector) == 32
    assert _norm(vector) == 0.0  # no tokens -> no signal, and no divide-by-zero


def test_shared_vocabulary_scores_higher_than_disjoint_text():
    provider = HashingEmbeddingProvider(dim=512)
    base = provider.embed(["revenue grew on strong product sales"])[0]
    overlapping = provider.embed(["product sales drove revenue"])[0]
    disjoint = provider.embed(["hurricane flooded the coastal harbor"])[0]
    assert _cosine(base, overlapping) > _cosine(base, disjoint)


def test_embed_chunks_fills_every_chunk():
    provider = HashingEmbeddingProvider(dim=64)
    chunks = [
        Chunk(
            document_id="doc",
            tenant="acme",
            doc_type="10-K",
            page_number=1,
            chunk_index=i,
            text=f"section {i} body text",
        )
        for i in range(5)
    ]
    embed_chunks(provider, chunks)
    assert all(chunk.embedding is not None and len(chunk.embedding) == 64 for chunk in chunks)


def test_embed_chunks_rejects_a_provider_that_drops_vectors():
    class DroppingProvider:
        dim = 64

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Simulates the production bug: fewer vectors than inputs.
            return [[0.0] * 64 for _ in texts[:-1]]

    chunks = [
        Chunk(document_id="d", tenant="t", doc_type="10-K", page_number=1, chunk_index=i, text="x")
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="count mismatch"):
        embed_chunks(DroppingProvider(), chunks)


def test_embed_chunks_on_empty_list_is_a_noop():
    assert embed_chunks(HashingEmbeddingProvider(dim=16), []) == []


def test_factory_builds_hash_provider_and_rejects_unknown():
    provider = get_embedding_provider(Settings(embedding_provider="hash", embedding_dim=48))
    assert isinstance(provider, HashingEmbeddingProvider)
    assert provider.dim == 48
    with pytest.raises(ValueError, match="unknown embedding_provider"):
        get_embedding_provider(Settings(embedding_provider="nope"))
