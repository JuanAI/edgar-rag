"""SQS message validation at the edge.

A real production lesson: validation that only checks a field is *present*
lets a message with empty strings pass and then fail deep in the
pipeline. Here we reject empty/whitespace-only required fields up front with a
clear error, so malformed messages die at the boundary, not three stages in.
"""
from __future__ import annotations

import json

from ..models import IngestionMessage

_REQUIRED_NON_EMPTY = ("document_id", "s3_bucket", "s3_key", "tenant")


class InvalidMessageError(ValueError):
    """The SQS body is missing or has empty required fields."""


def parse_message(body: str) -> IngestionMessage:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InvalidMessageError(f"body is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise InvalidMessageError("body must be a JSON object")

    for field in _REQUIRED_NON_EMPTY:
        value = data.get(field)
        # The bug fix: present-but-empty must be rejected, not just missing.
        if value is None or not str(value).strip():
            raise InvalidMessageError(f"required field {field!r} is missing or empty")

    return IngestionMessage.model_validate(data)
