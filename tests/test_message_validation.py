"""Edge validation (a real production lesson): present-but-empty required fields
must be rejected at the boundary, not just checked for presence."""

import json

import pytest

from edgar_rag.ingestion.messages import InvalidMessageError, parse_message


def _valid() -> dict:
    return {
        "document_id": "AAPL-10K-1",
        "s3_bucket": "edgar-documents",
        "s3_key": "AAPL/x/doc.htm",
        "tenant": "AAPL",
        "doc_type": "10-K",
    }


def test_valid_message_parses():
    msg = parse_message(json.dumps(_valid()))
    assert msg.tenant == "AAPL"
    assert msg.doc_type == "10-K"


@pytest.mark.parametrize("field", ["document_id", "s3_bucket", "s3_key", "tenant"])
def test_empty_string_is_rejected(field):
    bad = _valid() | {field: "   "}  # whitespace-only, not missing
    with pytest.raises(InvalidMessageError):
        parse_message(json.dumps(bad))


@pytest.mark.parametrize("field", ["document_id", "s3_bucket", "s3_key", "tenant"])
def test_missing_field_is_rejected(field):
    bad = _valid()
    del bad[field]
    with pytest.raises(InvalidMessageError):
        parse_message(json.dumps(bad))


def test_non_json_is_rejected():
    with pytest.raises(InvalidMessageError):
        parse_message("not-json{")
