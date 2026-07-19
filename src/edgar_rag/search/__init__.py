"""OpenSearch-backed search: index creation and bulk indexing (write path)."""

from __future__ import annotations

from .opensearch import KnnConfig, build_client, bulk_index, ensure_index, index_mapping

__all__ = [
    "KnnConfig",
    "build_client",
    "bulk_index",
    "ensure_index",
    "index_mapping",
]
