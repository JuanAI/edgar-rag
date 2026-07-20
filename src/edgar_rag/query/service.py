"""The retrieval read path: a question in, cited chunks out.

    question -> embed -> tenant-scoped kNN -> RetrievedChunks + Citations

This is retrieval only. Turning the retrieved chunks into a generated answer
(the "G" in RAG) and exposing an HTTP endpoint are later steps; keeping them
separate makes the tenant-scoped retrieval — the security-critical part — easy
to reason about and test on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch

from ..config import Settings
from ..embeddings import EmbeddingProvider, get_embedding_provider
from ..models import Citation, RetrievedChunk
from ..search.opensearch import build_client, knn_search


@dataclass
class RetrievalResult:
    """What a retrieval returns: the ranked chunks and their citations."""

    chunks: list[RetrievedChunk]
    citations: list[Citation]


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(document_id=c.document_id, page_number=c.page_number, score=c.score)
        for c in chunks
    ]


def retrieve(
    settings: Settings,
    *,
    question: str,
    tenant: str,
    doc_type: str | None = None,
    k: int | None = None,
    embedder: EmbeddingProvider | None = None,
    client: OpenSearch | None = None,
) -> RetrievalResult:
    """Embed the question and return the nearest chunks for `tenant`, with
    citations.

    `embedder` and `client` can be injected (tests pass fakes); by default they
    are built from settings. The question is embedded with the SAME provider used
    at ingestion, so query and chunk vectors live in the same space.
    """
    embedder = embedder or get_embedding_provider(settings)
    client = client if client is not None else build_client(settings)
    vector = embedder.embed([question])[0]
    chunks = knn_search(
        client,
        settings.opensearch_index,
        vector=vector,
        tenant=tenant,
        doc_type=doc_type,
        k=k or settings.top_k,
    )
    return RetrievalResult(chunks=chunks, citations=_citations(chunks))
