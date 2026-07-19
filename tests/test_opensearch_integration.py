"""Integration tests for the OpenSearch write path against a real cluster.

These prove the index actually accepts our mapping and our documents — the parts
a unit test cannot. They auto-skip unless a cluster is reachable on localhost, so
`make test` stays green without infra; run `make up` first (or `make
test-integration`) to exercise them.

Each test uses a unique index name and deletes it afterwards, so runs never
collide and leave nothing behind.
"""

from uuid import uuid4

import pytest

from edgar_rag.config import Settings
from edgar_rag.models import Chunk
from edgar_rag.search.opensearch import KnnConfig, build_client, bulk_index, ensure_index


def _settings() -> Settings:
    # Tests run on the host, so reach the cluster at localhost rather than the
    # in-network "opensearch" service name used inside docker compose.
    return Settings(opensearch_host="localhost")


@pytest.fixture
def client():
    settings = _settings()
    os_client = build_client(settings)
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


def _chunk(chunk_index: int, embedding: list[float]) -> Chunk:
    return Chunk(
        document_id="AAPL-10K-2023",
        tenant="acme",
        doc_type="10-K",
        page_number=1,
        chunk_index=chunk_index,
        text=f"chunk {chunk_index}",
        embedding=embedding,
    )


def test_ensure_index_is_idempotent(client, index):
    # The fixture already created it, so a second call must be a no-op.
    assert ensure_index(client, index, KnnConfig(dim=3)) is False


def test_bulk_index_writes_every_chunk_and_is_retrievable(client, index):
    chunks = [
        _chunk(0, [1.0, 0.0, 0.0]),
        _chunk(1, [0.0, 1.0, 0.0]),
        _chunk(2, [0.0, 0.0, 1.0]),
    ]
    indexed = bulk_index(client, index, chunks)
    assert indexed == 3

    # refresh=True made them immediately searchable; verify by stable id.
    doc = client.get(index=index, id="AAPL-10K-2023:1:1")
    assert doc["_source"]["text"] == "chunk 1"
    assert doc["_source"]["embedding"] == [0.0, 1.0, 0.0]
    assert client.count(index=index)["count"] == 3


def test_reindexing_same_chunk_upserts_in_place(client, index):
    bulk_index(client, index, [_chunk(0, [1.0, 0.0, 0.0])])
    bulk_index(client, index, [_chunk(0, [0.0, 0.0, 1.0])])  # same id, new vector
    # update + doc_as_upsert merges into the existing doc: one document, new vector.
    assert client.count(index=index)["count"] == 1
    doc = client.get(index=index, id="AAPL-10K-2023:1:0")
    assert doc["_source"]["embedding"] == [0.0, 0.0, 1.0]
