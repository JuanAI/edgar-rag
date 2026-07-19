"""Chunk a document's pages into overlapping, token-sized passages.

Chunking is done by *tokens* (via tiktoken), not characters, so each chunk's
cost against the LLM context window is predictable regardless of how dense the
text is — prose and number/table-heavy filing text tokenize very differently,
and character windows give wildly varying token counts. This mirrors a
production token-based chunker.

Each page can yield MANY chunks. This is the shape that surfaced a real
production bug: embeddings must be List[List[float]] (one vector per chunk per
page), not one vector per page — otherwise chunks silently drop out of search.
`chunk_pages` guarantees every chunk is emitted; a test asserts the count.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from ..models import Chunk, Page


@lru_cache(maxsize=4)
def _get_encoding(name: str) -> tiktoken.Encoding:
    # Cached: building an encoding is relatively expensive and thread-safe to share.
    return tiktoken.get_encoding(name)


def _token_windows(
    text: str, encoding: tiktoken.Encoding, size_tokens: int, overlap_percentage: float
) -> list[tuple[str, int, int]]:
    """Return (chunk_text, token_count, char_offset) sliding windows over `text`."""
    text = text.strip()
    if not text:
        return []
    if size_tokens <= 0:
        raise ValueError("chunk size (tokens) must be positive")
    if not 0.0 <= overlap_percentage < 1.0:
        raise ValueError("overlap_percentage must be in [0.0, 1.0)")

    tokens = encoding.encode(text)
    # Character offset of each token within `text`, so a chunk can be traced
    # back to its exact position in the source page.
    _, token_offsets = encoding.decode_with_offsets(tokens)

    overlap_tokens = int(size_tokens * overlap_percentage)
    step = max(1, size_tokens - overlap_tokens)

    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens).strip()
        if chunk_text:
            windows.append((chunk_text, len(chunk_tokens), token_offsets[start]))
        start += step
    return windows


def chunk_pages(
    pages: list[Page],
    *,
    document_id: str,
    tenant: str,
    doc_type: str,
    size_tokens: int,
    overlap_percentage: float,
    encoding_name: str = "cl100k_base",
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    encoding = _get_encoding(encoding_name)
    chunks: list[Chunk] = []
    for page in pages:
        windows = _token_windows(page.text, encoding, size_tokens, overlap_percentage)
        for idx, (piece, token_count, char_offset) in enumerate(windows):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    tenant=tenant,
                    doc_type=doc_type,
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=piece,
                    token_count=token_count,
                    char_offset=char_offset,
                    metadata=metadata or {},
                )
            )
    return chunks
