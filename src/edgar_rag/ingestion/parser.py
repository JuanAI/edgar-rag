"""Turn a raw EDGAR filing (HTML or text) into logical 'pages' (sections).

EDGAR primary documents are typically large HTML files with no real pages. We
strip markup, then group the text into sections sized by *tokens* (via tiktoken,
the same encoding used for chunking) so the whole pipeline speaks one unit and
page sizes stay predictable regardless of text density. Each page yields many
chunks downstream, and its page number gives retrieved chunks something to cite.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

from ..models import Page

_WHITESPACE = re.compile(r"[ \t]+")


@lru_cache(maxsize=4)
def _get_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def _html_to_text(raw: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def lint_text(text: str) -> str:
    """Light token-reduction cleanup for parsed text.

    Caps long runs of repeated punctuation — the dotted "leaders" and rule
    lines in 10-K tables of contents and tables — which are pure token waste.
    (A production PDF pipeline lints more aggressively: de-hyphenating
    line-broken words and rejoining wrapped sentences; HTML arrives cleaner, so
    those rules aren't needed here.)
    """
    # "Item 1......................5" -> "Item 1...5"
    return re.sub(r"([^\w\s])\1{3,}", r"\1\1\1", text)


def parse_document(
    raw_bytes: bytes,
    *,
    section_target_tokens: int = 500,
    encoding_name: str = "cl100k_base",
) -> list[Page]:
    raw = raw_bytes.decode("utf-8", errors="replace")
    is_html = "<html" in raw[:2000].lower() or "<div" in raw[:5000].lower()
    text = _html_to_text(raw) if is_html else raw
    text = lint_text(text)

    encoding = _get_encoding(encoding_name)

    # Normalise whitespace, drop blank lines.
    lines = [_WHITESPACE.sub(" ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Group lines into sections by token count (the same unit as chunking).
    sections: list[str] = []
    buffer: list[str] = []
    tokens = 0
    for line in lines:
        buffer.append(line)
        tokens += len(encoding.encode(line))
        if tokens >= section_target_tokens:
            sections.append("\n".join(buffer))
            buffer, tokens = [], 0
    if buffer:
        sections.append("\n".join(buffer))

    return [Page(page_number=i + 1, text=s) for i, s in enumerate(sections) if s.strip()]
