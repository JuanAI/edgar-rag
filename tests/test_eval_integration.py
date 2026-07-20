"""End-to-end retrieval evaluation against a real cluster with real MiniLM.

This is both a demonstration (prints/produces the metrics) and a regression gate:
if a change degrades retrieval below the thresholds, it fails. Auto-skips unless
the `local` extra is installed AND OpenSearch is reachable on localhost.
"""

import pytest

pytest.importorskip("sentence_transformers")

from edgar_rag.config import Settings  # noqa: E402
from edgar_rag.eval import run_eval  # noqa: E402
from edgar_rag.search.opensearch import build_client  # noqa: E402


def _settings() -> Settings:
    return Settings(opensearch_host="localhost")


@pytest.fixture(scope="module")
def _require_cluster():
    try:
        build_client(_settings()).info()
    except Exception:
        pytest.skip("OpenSearch not reachable on localhost:9200 — run `make up` first")


def test_retrieval_meets_quality_thresholds(_require_cluster):
    report = run_eval(_settings(), k=5)

    assert report.num_queries == 8
    # Conservative gates: the labeled answer page should reliably surface near the
    # top for these distinct-topic questions. Tighten as the corpus grows.
    assert report.mean_recall_at_k >= 0.85, report
    assert report.mean_ndcg_at_k >= 0.70, report
    assert report.mrr >= 0.60, report


def test_every_query_finds_its_answer_somewhere_in_top_k(_require_cluster):
    report = run_eval(_settings(), k=5)
    missed = [s.question for s in report.per_query if s.recall_at_k == 0.0]
    assert not missed, f"queries with no relevant hit in top-5: {missed}"
