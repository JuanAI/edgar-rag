from edgar_rag.ingestion.chunker import chunk_pages
from edgar_rag.models import Page


def _pages():
    return [
        Page(page_number=1, text="alpha " * 400),  # long -> multiple chunks
        Page(page_number=2, text="beta gamma delta"),  # short -> one chunk
    ]


def _chunk(pages, size_tokens=64, overlap_percentage=0.2):
    return chunk_pages(
        pages,
        document_id="D1",
        tenant="T1",
        doc_type="10-K",
        size_tokens=size_tokens,
        overlap_percentage=overlap_percentage,
    )


def test_long_page_produces_multiple_chunks():
    page1 = [c for c in _chunk(_pages()) if c.page_number == 1]
    assert len(page1) > 1  # a single page yields many chunks


def test_chunk_ids_are_unique():
    ids = [c.chunk_id for c in _chunk(_pages())]
    assert len(ids) == len(set(ids))


def test_token_count_is_populated_and_bounded():
    chunks = _chunk(_pages(), size_tokens=64)
    assert all(c.token_count is not None for c in chunks)
    assert all(0 < c.token_count <= 64 for c in chunks)


def test_char_offset_is_monotonic_within_a_page():
    offsets = [c.char_offset for c in _chunk(_pages()) if c.page_number == 1]
    assert all(o is not None for o in offsets)
    assert offsets == sorted(offsets)  # windows advance through the text


def test_overlap_shares_content_between_consecutive_chunks():
    # distinct tokens so we can detect a shared span across a boundary
    text = " ".join(f"word{i}" for i in range(300))
    chunks = [
        c for c in _chunk([Page(page_number=1, text=text)], size_tokens=50) if c.page_number == 1
    ]
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words  # overlap region is shared


def test_invalid_overlap_percentage_rejected():
    import pytest

    with pytest.raises(ValueError):
        _chunk(_pages(), overlap_percentage=1.0)
