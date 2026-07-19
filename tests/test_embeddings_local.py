"""Integration tests for the real MiniLM (sentence-transformers) backend.

These download and run the actual production embedder, so they are the proof
that the real path works — not just the plumbing. They auto-skip unless the
`local` extra is installed (so `make test` on the dev extra alone stays fast);
CI installs the extra and caches the model, so they run for real on every PR.
"""

import math

import pytest

pytest.importorskip("sentence_transformers")

from edgar_rag.config import Settings  # noqa: E402
from edgar_rag.embeddings import embed_chunks, get_embedding_provider  # noqa: E402
from edgar_rag.ingestion.chunker import chunk_pages  # noqa: E402
from edgar_rag.models import Page  # noqa: E402

# all-MiniLM-L6-v2 emits 384-dimensional vectors.
_MINILM_DIM = 384


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_local_embeddings_have_expected_dim_and_unit_norm():
    provider = get_embedding_provider(Settings(embedding_provider="local"))
    vectors = provider.embed(["Risk factors and competition."])
    assert provider.dim == _MINILM_DIM
    assert len(vectors) == 1
    assert len(vectors[0]) == _MINILM_DIM
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0, rel_tol=1e-5)


def test_local_embeddings_capture_meaning_not_just_words():
    provider = get_embedding_provider(Settings(embedding_provider="local"))
    base, paraphrase, unrelated = provider.embed(
        [
            "The company's quarterly revenue grew on strong product sales.",
            "Sales rose sharply, lifting revenue for the quarter.",
            "A hurricane flooded the coastal harbor overnight.",
        ]
    )
    # The paraphrase shares little vocabulary but the same meaning; a real
    # semantic model must score it closer than the unrelated sentence.
    assert _cosine(base, paraphrase) > _cosine(base, unrelated)


def test_local_embeds_real_chunks_end_to_end():
    pages = [Page(page_number=1, text="Item 1A. Risk Factors. " * 80)]
    chunks = chunk_pages(
        pages,
        document_id="D",
        tenant="T",
        doc_type="10-K",
        size_tokens=64,
        overlap_percentage=0.2,
    )
    provider = get_embedding_provider(Settings(embedding_provider="local"))
    embed_chunks(provider, chunks)
    assert chunks
    assert all(c.embedding is not None and len(c.embedding) == provider.dim for c in chunks)
