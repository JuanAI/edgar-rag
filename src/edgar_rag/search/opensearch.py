"""OpenSearch write path: connect, create the vector index, and bulk-index chunks.

This is the second half of ingestion — after chunks are embedded, they land here
to become searchable. The index stores each chunk's `embedding` as a `knn_vector`
so nearest-neighbour (kNN) search can later find the chunks closest to a question.
The read path (the kNN query itself) lives in the query step, not here.

Multi-tenancy is by shared index: every chunk carries a `tenant` field and the
query path always filters on it, so a tenant can never retrieve another's data.

Production lessons baked in:
  * the vector field's `dimension` must match the embedder, so the index is
    built from a KnnConfig rather than hard-coded values;
  * a chunk with no embedding is invisible to kNN — we refuse it loudly rather
    than silently index an unsearchable document; and
  * writes are upserts (update + doc_as_upsert), so a later stage can enrich the
    same chunk document without a first write clobbering it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import OpenSearchException

from ..config import Settings
from ..models import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnnConfig:
    """Parameters for the kNN vector index.

    `m` and `ef_construction` are HNSW's graph knobs: higher values improve
    recall at the cost of index build time and memory. The defaults mirror
    common production settings and can be overridden from Settings.
    """

    dim: int
    engine: str = "lucene"
    space_type: str = "cosinesimil"
    m: int = 16
    ef_construction: int = 128

    @classmethod
    def from_settings(cls, settings: Settings) -> KnnConfig:
        return cls(
            dim=settings.embedding_dim,
            engine=settings.opensearch_engine,
            space_type=settings.opensearch_space_type,
            m=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
        )


def build_client(settings: Settings) -> OpenSearch:
    """Open a client from settings. Security is disabled for the local cluster,
    so there is no auth; flip use_ssl/host to point at a secured deployment."""
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_compress=True,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )


def index_mapping(knn: KnnConfig) -> dict[str, Any]:
    """The index definition: scalar fields for filtering plus the kNN vector.

    `knn: True` turns on approximate nearest-neighbour search. Metadata is stored
    as a single `flat_object` so arbitrary per-document keys are searchable
    (via `metadata.<key>`) without expanding the mapping or colliding with the
    trusted top-level fields.
    """
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "tenant": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "token_count": {"type": "integer"},
                "char_offset": {"type": "integer"},
                "text": {"type": "text"},
                "metadata": {"type": "flat_object"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": knn.dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": knn.space_type,
                        "engine": knn.engine,
                        "parameters": {"ef_construction": knn.ef_construction, "m": knn.m},
                    },
                },
            }
        },
    }


def ensure_index(client: OpenSearch, index: str, knn: KnnConfig) -> bool:
    """Create the index if it does not exist. Returns True only if it created it,
    so callers can tell a first-time setup from a no-op (idempotent)."""
    try:
        if client.indices.exists(index=index):
            logger.info("index already exists", extra={"index": index})
            return False
        client.indices.create(index=index, body=index_mapping(knn))
        logger.info("created index", extra={"index": index, "dim": knn.dim})
        return True
    except OpenSearchException:
        logger.exception("failed to ensure index", extra={"index": index})
        raise


def _chunk_to_action(index: str, chunk: Chunk) -> dict[str, Any]:
    """Turn one chunk into a bulk upsert action.

    `_id` is the stable chunk_id, and the op is `update` with `doc_as_upsert`, so
    re-indexing a chunk merges in place (no duplicates) and a later co-writer's
    fields survive a re-ingest instead of being overwritten.
    """
    if chunk.embedding is None:
        raise ValueError(f"chunk {chunk.chunk_id} has no embedding")
    return {
        "_op_type": "update",
        "_index": index,
        "_id": chunk.chunk_id,
        "doc": {
            "document_id": chunk.document_id,
            "tenant": chunk.tenant,
            "doc_type": chunk.doc_type,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_offset": chunk.char_offset,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata,
        },
        "doc_as_upsert": True,
    }


def bulk_index(client: OpenSearch, index: str, chunks: list[Chunk]) -> int:
    """Upsert all chunks in one bulk request; returns the number written.

    `refresh=True` makes the documents searchable immediately — convenient for a
    demo and for tests that index then query in the same breath. A bulk failure
    is logged with its count and re-raised rather than silently swallowed.
    """
    actions = [_chunk_to_action(index, chunk) for chunk in chunks]
    if not actions:
        return 0
    try:
        success, errors = helpers.bulk(client, actions, refresh=True)
    except helpers.BulkIndexError as exc:
        logger.error(
            "bulk index failed",
            extra={"index": index, "errors": len(exc.errors), "attempted": len(actions)},
        )
        raise
    logger.info("bulk indexed chunks", extra={"index": index, "count": success})
    return success
