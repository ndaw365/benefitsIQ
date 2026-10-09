"""Split pages into overlapping fixed-size character windows."""

from dataclasses import dataclass

from src.rag.load import Page


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>:p<page>:c<n>", stable across runs for the same config
    doc_id: str
    page: int
    start_char: int  # offset within the page text
    text: str


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Return (start_offset, window) pairs covering the whole text.

    Each window starts `size - overlap` characters after the previous one, so
    neighbouring windows share `overlap` characters and a sentence cut at one
    boundary appears whole in the next window.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    step = size - overlap
    windows = []
    for start in range(0, len(text), step):
        windows.append((start, text[start : start + size]))
        if start + size >= len(text):  # this window already reaches the end
            break
    return windows


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Chunk each page separately, so every chunk cites exactly one page."""
    return [
        Chunk(f"{p.doc_id}:p{p.page}:c{n}", p.doc_id, p.page, start, text)
        for p in pages
        for n, (start, text) in enumerate(chunk_text(p.text, size, overlap))
    ]
