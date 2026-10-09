"""Tests for chunking (pure Python, no models)."""

import pytest

from src.rag.chunk import chunk_pages, chunk_text
from src.rag.load import Page

TEXT = "".join(chr(ord("a") + i % 26) for i in range(1000))


def test_windows_cover_all_text() -> None:
    windows = chunk_text(TEXT, size=300, overlap=50)
    rebuilt = windows[0][1] + "".join(w[50:] for _, w in windows[1:])
    assert rebuilt == TEXT


def test_neighbours_share_exactly_overlap_chars() -> None:
    windows = chunk_text(TEXT, size=300, overlap=50)
    for (_, a), (_, b) in zip(windows, windows[1:]):
        assert a[-50:] == b[:50]


def test_offsets_point_at_the_window_text() -> None:
    for start, window in chunk_text(TEXT, size=300, overlap=50):
        assert TEXT[start : start + len(window)] == window


def test_no_tiny_trailing_window() -> None:
    # 600 chars, size 300, overlap 0 -> exactly 2 windows, not a third empty one
    assert len(chunk_text("x" * 600, size=300, overlap=0)) == 2


def test_short_text_is_one_window() -> None:
    assert chunk_text("short", size=300, overlap=50) == [(0, "short")]


def test_empty_text_gives_no_windows() -> None:
    assert chunk_text("", size=300, overlap=50) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_bad_parameters_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)


def test_chunk_ids_are_unique_and_carry_page() -> None:
    pages = [Page("doc", 1, TEXT), Page("doc", 2, TEXT)]
    chunks = chunk_pages(pages, size=300, overlap=50)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[0].chunk_id == "doc:p1:c0"
