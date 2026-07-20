"""OpenSearch-backed search: index creation, bulk indexing, and kNN retrieval."""

from __future__ import annotations

from .opensearch import (
    KnnConfig,
    build_client,
    build_knn_query,
    bulk_index,
    ensure_index,
    index_mapping,
    knn_search,
)

__all__ = [
    "KnnConfig",
    "build_client",
    "build_knn_query",
    "bulk_index",
    "ensure_index",
    "index_mapping",
    "knn_search",
]
