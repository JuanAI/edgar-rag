"""Embedding providers and the batch helper the ingestion path calls.

`get_embedding_provider` selects a backend from settings; `embed_chunks` runs a
batch of chunks through whichever provider is wired in and — crucially —
guarantees every chunk comes back with a vector.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Chunk
from .base import EmbeddingProvider
from .hashing import HashingEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "embed_chunks",
    "get_embedding_provider",
]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider named by `settings.embedding_provider`."""
    provider = settings.embedding_provider
    if provider == "local":
        # Imported lazily: sentence-transformers is an optional dependency.
        from .local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(settings.embedding_model)
    if provider == "hash":
        return HashingEmbeddingProvider(settings.embedding_dim)
    raise ValueError(
        f"unknown embedding_provider {provider!r}; supported providers: 'local', 'hash'"
    )


def embed_chunks(provider: EmbeddingProvider, chunks: list[Chunk]) -> list[Chunk]:
    """Embed every chunk in place and return the same list.

    The `len` check is not paranoia: the shape here (one vector per chunk) is
    exactly where a real production bug hid — a provider that returned one
    vector per *page* left later chunks with `embedding=None`, so they silently
    vanished from search. We fail loudly instead.
    """
    if not chunks:
        return chunks
    vectors = provider.embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(
            f"embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks"
        )
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
    return chunks
