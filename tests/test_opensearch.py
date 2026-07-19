"""Fast unit tests for the OpenSearch write path — the pure logic that needs no
live cluster: the index mapping and how a chunk becomes a bulk action. The real
cluster round-trip is covered in test_opensearch_integration.py.
"""

import pytest

from edgar_rag.config import Settings
from edgar_rag.models import Chunk
from edgar_rag.search.opensearch import KnnConfig, _chunk_to_action, index_mapping


def _chunk(metadata: dict[str, str] | None = None) -> Chunk:
    return Chunk(
        document_id="AAPL-10K-2023",
        tenant="acme",
        doc_type="10-K",
        page_number=2,
        chunk_index=5,
        text="Risk factors and competition.",
        token_count=6,
        char_offset=120,
        embedding=[0.1, 0.2, 0.3],
        metadata=metadata or {},
    )


def test_knn_config_from_settings_reads_the_tuning_knobs():
    knn = KnnConfig.from_settings(Settings(embedding_dim=384, hnsw_m=24, hnsw_ef_construction=200))
    assert knn.dim == 384
    assert knn.m == 24
    assert knn.ef_construction == 200
    assert knn.engine == "lucene"
    assert knn.space_type == "cosinesimil"


def test_index_mapping_declares_a_tuned_knn_vector():
    mapping = index_mapping(KnnConfig(dim=384, m=24, ef_construction=200))
    assert mapping["settings"]["index"]["knn"] is True
    embedding = mapping["mappings"]["properties"]["embedding"]
    assert embedding["type"] == "knn_vector"
    assert embedding["dimension"] == 384
    assert embedding["method"]["space_type"] == "cosinesimil"
    assert embedding["method"]["parameters"] == {"ef_construction": 200, "m": 24}
    # Metadata is a flat_object so arbitrary keys stay searchable and namespaced.
    assert mapping["mappings"]["properties"]["metadata"]["type"] == "flat_object"


def test_chunk_action_is_an_upsert_with_stable_id_and_the_vector():
    action = _chunk_to_action("edgar-chunks", _chunk())
    # update + doc_as_upsert merges in place; _id is the chunk_id so re-indexing
    # never duplicates.
    assert action["_op_type"] == "update"
    assert action["doc_as_upsert"] is True
    assert action["_id"] == "AAPL-10K-2023:2:5"
    assert action["_index"] == "edgar-chunks"
    assert action["doc"]["embedding"] == [0.1, 0.2, 0.3]
    assert action["doc"]["tenant"] == "acme"


def test_chunk_metadata_is_stored_as_a_nested_object():
    action = _chunk_to_action("edgar-chunks", _chunk(metadata={"cik": "0000320193"}))
    assert action["doc"]["metadata"] == {"cik": "0000320193"}


def test_chunk_without_embedding_is_refused():
    chunk = Chunk(
        document_id="D",
        tenant="t",
        doc_type="10-K",
        page_number=1,
        chunk_index=0,
        text="x",
    )
    with pytest.raises(ValueError, match="has no embedding"):
        _chunk_to_action("edgar-chunks", chunk)
