"""Fast unit tests for the read path — the tenant filter and result parsing,
with no live cluster. The real cluster round-trip and tenant isolation are in
test_query_integration.py.
"""

from typing import Any, cast

from opensearchpy import OpenSearch

from edgar_rag.config import Settings
from edgar_rag.query import retrieve
from edgar_rag.search.opensearch import build_knn_query, knn_search


def test_build_knn_query_always_filters_by_tenant():
    body = build_knn_query([0.1, 0.2, 0.3], tenant="acme", doc_type=None, k=5)
    filters = body["query"]["knn"]["embedding"]["filter"]["bool"]["must"]
    assert {"term": {"tenant": "acme"}} in filters
    # No doc_type scope requested -> tenant is the only filter.
    assert len(filters) == 1
    assert body["size"] == 5
    assert body["query"]["knn"]["embedding"]["k"] == 5


def test_build_knn_query_adds_doc_type_when_scoped():
    body = build_knn_query([0.1], tenant="acme", doc_type="10-K", k=3)
    filters = body["query"]["knn"]["embedding"]["filter"]["bool"]["must"]
    assert {"term": {"tenant": "acme"}} in filters
    assert {"term": {"doc_type": "10-K"}} in filters


class _FakeClient:
    """Captures the query body and returns a canned OpenSearch response."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.last_body: dict[str, Any] | None = None

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.last_body = body
        return {"hits": {"hits": self.hits}}


def _hit(text: str, score: float) -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {
            "text": text,
            "document_id": "AAPL-10K-2023",
            "doc_type": "10-K",
            "page_number": 4,
        },
    }


def test_knn_search_parses_hits_into_retrieved_chunks():
    client = _FakeClient([_hit("risk factors", 0.91), _hit("competition", 0.80)])
    results = knn_search(
        cast(OpenSearch, client), "edgar-chunks", vector=[0.1], tenant="acme", doc_type=None, k=2
    )
    assert [r.text for r in results] == ["risk factors", "competition"]
    assert results[0].score == 0.91  # OpenSearch _score becomes the similarity score
    assert results[0].document_id == "AAPL-10K-2023"


class _FakeEmbedder:
    name = "fake"
    dim = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_retrieve_embeds_question_scopes_to_tenant_and_builds_citations():
    client = _FakeClient([_hit("Apple faces competition.", 0.88)])
    result = retrieve(
        Settings(),
        question="What are the risks?",
        tenant="acme",
        embedder=_FakeEmbedder(),
        client=cast(OpenSearch, client),
    )
    # The retrieval was scoped to the caller's tenant.
    filters = client.last_body["query"]["knn"]["embedding"]["filter"]["bool"]["must"]
    assert {"term": {"tenant": "acme"}} in filters
    # Citations are derived from the retrieved chunks.
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "AAPL-10K-2023"
    assert result.citations[0].page_number == 4
    assert result.citations[0].score == 0.88


def test_retrieve_defaults_k_to_settings_top_k():
    client = _FakeClient([])
    retrieve(
        Settings(top_k=7),
        question="q",
        tenant="acme",
        embedder=_FakeEmbedder(),
        client=cast(OpenSearch, client),
    )
    assert client.last_body["size"] == 7
