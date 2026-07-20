"""Integration tests for tenant-scoped kNN retrieval against a real cluster.

The headline test proves the security property: a tenant's query never returns
another tenant's chunks — even when the other tenant's vector is a perfect match.
Auto-skips unless OpenSearch is reachable on localhost; run `make up` first.
"""

from uuid import uuid4

import pytest

from edgar_rag.config import Settings
from edgar_rag.models import Chunk
from edgar_rag.search.opensearch import (
    KnnConfig,
    build_client,
    bulk_index,
    ensure_index,
    knn_search,
)


def _settings() -> Settings:
    return Settings(opensearch_host="localhost")


@pytest.fixture
def client():
    os_client = build_client(_settings())
    try:
        os_client.info()
    except Exception:
        pytest.skip("OpenSearch not reachable on localhost:9200 — run `make up` first")
    return os_client


@pytest.fixture
def index(client):
    name = f"edgar-test-{uuid4().hex[:8]}"
    ensure_index(client, name, KnnConfig(dim=3))
    yield name
    client.indices.delete(index=name, ignore=[404])


def _chunk(tenant: str, chunk_index: int, text: str, embedding: list[float]) -> Chunk:
    return Chunk(
        document_id=f"{tenant}-doc",
        tenant=tenant,
        doc_type="10-K",
        page_number=1,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding,
    )


def test_retrieval_never_crosses_tenants(client, index):
    # globex owns the vector [0,1,0]; acme owns [1,0,0].
    bulk_index(
        client,
        index,
        [
            _chunk("acme", 0, "acme risk factors", [1.0, 0.0, 0.0]),
            _chunk("globex", 0, "globex risk factors", [0.0, 1.0, 0.0]),
        ],
    )

    # Query as acme with a vector IDENTICAL to globex's chunk. A pure similarity
    # search would return globex's chunk first; the tenant filter must exclude it.
    hits = knn_search(client, index, vector=[0.0, 1.0, 0.0], tenant="acme", doc_type=None, k=5)
    assert hits, "acme should still get its own chunk back"
    assert {h.document_id for h in hits} == {"acme-doc"}
    assert all("globex" not in h.text for h in hits)


def test_retrieval_ranks_nearest_first_within_a_tenant(client, index):
    bulk_index(
        client,
        index,
        [
            _chunk("acme", 0, "near", [1.0, 0.0, 0.0]),
            _chunk("acme", 1, "far", [0.0, 0.0, 1.0]),
        ],
    )
    hits = knn_search(client, index, vector=[1.0, 0.0, 0.0], tenant="acme", doc_type=None, k=2)
    assert [h.text for h in hits] == ["near", "far"]  # nearest by cosine first
