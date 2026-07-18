import tiktoken

from edgar_rag.ingestion.parser import lint_text, parse_document

_ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_ENC.encode(text))


def test_html_filing_produces_multiple_pages():
    html = (
        b"<html><body>"
        + b"".join((f"<div>Item {i}. Section {i} body text. " * 40).encode() for i in range(1, 6))
        + b"</body></html>"
    )
    pages = parse_document(html, section_target_tokens=200)
    assert len(pages) >= 3
    assert all(p.text.strip() for p in pages)
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))


def test_pages_are_token_sized():
    # Every full section reaches the target; only the trailing remainder may be short.
    pages = parse_document(b"Risk factors and competition. \n" * 500, section_target_tokens=200)
    assert len(pages) >= 3
    for p in pages[:-1]:
        assert _tokens(p.text) >= 200


def test_plaintext_is_parsed_without_html():
    pages = parse_document(b"Just plain text.\n" * 100, section_target_tokens=100)
    assert len(pages) >= 1
    assert "Just plain text." in pages[0].text


def test_script_and_style_are_stripped():
    html = b"<html><body><script>var x=1;</script><p>Real filing text here.</p></body></html>"
    joined = " ".join(p.text for p in parse_document(html))
    assert "Real filing text" in joined
    assert "var x=1" not in joined


def test_lint_text_caps_dot_leaders_and_saves_tokens():
    dirty = "Item 1. Business" + "." * 40 + "5"
    clean = lint_text(dirty)
    assert "." * 40 not in clean  # the long dotted leader is gone
    assert "..." in clean  # capped, not fully removed
    assert _tokens(clean) < _tokens(dirty)  # linting reduces token count
