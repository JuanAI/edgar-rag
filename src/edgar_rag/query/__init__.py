"""Query (read) path: tenant-scoped retrieval of cited chunks."""

from __future__ import annotations

from .service import RetrievalResult, retrieve

__all__ = ["RetrievalResult", "retrieve"]
