"""Run the retrieval evaluation end to end and report aggregate metrics.

    ingest the labeled corpus -> for each query: embed -> kNN -> score vs qrels
    -> aggregate recall@k / nDCG@k / MRR

Uses a throwaway index (created and deleted per run) so it never touches real
data. The index vector dimension is taken from the embedder, so it always
matches whatever backend is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from opensearchpy import OpenSearch

from ..config import Settings
from ..embeddings import EmbeddingProvider, embed_chunks, get_embedding_provider
from ..ingestion.chunker import chunk_pages
from ..models import Page, RetrievedChunk
from ..search.opensearch import (
    KnnConfig,
    build_client,
    bulk_index,
    ensure_index,
    knn_search,
)
from .dataset import EvalDataset, EvalQuery, load_dataset
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass
class QueryScore:
    question: str
    recall_at_k: float
    ndcg_at_k: float
    reciprocal_rank: float


@dataclass
class EvalReport:
    k: int
    num_queries: int
    mean_recall_at_k: float
    mean_ndcg_at_k: float
    mrr: float
    per_query: list[QueryScore]


def _ingest(
    client: OpenSearch,
    index: str,
    embedder: EmbeddingProvider,
    dataset: EvalDataset,
    size_tokens: int,
    overlap_percentage: float,
) -> None:
    for doc in dataset.documents:
        pages = [Page(page_number=p.page_number, text=p.text) for p in doc.pages]
        chunks = chunk_pages(
            pages,
            document_id=doc.document_id,
            tenant=dataset.tenant,
            doc_type=doc.doc_type,
            size_tokens=size_tokens,
            overlap_percentage=overlap_percentage,
        )
        embed_chunks(embedder, chunks)
        bulk_index(client, index, chunks)


def _score_query(query: EvalQuery, retrieved: list[RetrievedChunk], k: int) -> QueryScore:
    relevant = {j.key: j.grade for j in query.relevant}
    ranked_keys = [(c.document_id, c.page_number) for c in retrieved]
    ranked_gains = [float(relevant.get(key, 0)) for key in ranked_keys]
    return QueryScore(
        question=query.question,
        recall_at_k=recall_at_k(ranked_keys, set(relevant), k),
        ndcg_at_k=ndcg_at_k(ranked_gains, [float(g) for g in relevant.values()], k),
        reciprocal_rank=reciprocal_rank(ranked_gains),
    )


def run_eval(
    settings: Settings,
    dataset: EvalDataset | None = None,
    *,
    k: int = 5,
    embedder: EmbeddingProvider | None = None,
    client: OpenSearch | None = None,
    size_tokens: int = 256,
    overlap_percentage: float = 0.1,
) -> EvalReport:
    """Ingest the corpus into a fresh index, run every query, and aggregate."""
    dataset = dataset or load_dataset()
    embedder = embedder or get_embedding_provider(settings)
    os_client = client if client is not None else build_client(settings)
    index = f"edgar-eval-{uuid4().hex[:8]}"

    knn = KnnConfig(
        dim=embedder.dim,
        engine=settings.opensearch_engine,
        space_type=settings.opensearch_space_type,
        m=settings.hnsw_m,
        ef_construction=settings.hnsw_ef_construction,
    )
    ensure_index(os_client, index, knn)
    try:
        _ingest(os_client, index, embedder, dataset, size_tokens, overlap_percentage)
        scores = []
        for query in dataset.queries:
            vector = embedder.embed([query.question])[0]
            hits = knn_search(
                os_client,
                index,
                vector=vector,
                tenant=dataset.tenant,
                doc_type=None,
                k=k,
            )
            scores.append(_score_query(query, hits, k))
    finally:
        os_client.indices.delete(index=index, ignore=[404])

    n = len(scores)
    return EvalReport(
        k=k,
        num_queries=n,
        mean_recall_at_k=sum(s.recall_at_k for s in scores) / n,
        mean_ndcg_at_k=sum(s.ndcg_at_k for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        per_query=scores,
    )
