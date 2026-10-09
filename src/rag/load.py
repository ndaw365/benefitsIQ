"""Load PDFs into page-level text records, keeping page numbers for citations."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

DOCS_DIR = Path("data/docs")
MIN_PAGE_CHARS = 50  # below this a page is blank or a cover; nothing worth retrieving


@dataclass(frozen=True)
class Page:
    doc_id: str  # file name without .pdf, e.g. "lincoln_moravian_ltd"
    page: int  # 1-based, matches the PDF viewer
    text: str


def clean(text: str) -> str:
    """Collapse whitespace; PDF extraction leaves line breaks mid-sentence."""
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: Path) -> list[Page]:
    """Extract one Page per PDF page, skipping near-empty pages."""
    pages = []
    for number, pdf_page in enumerate(PdfReader(path).pages, start=1):
        text = clean(pdf_page.extract_text() or "")
        if len(text) >= MIN_PAGE_CHARS:
            pages.append(Page(doc_id=path.stem, page=number, text=text))
    return pages


def load_all(docs_dir: Path = DOCS_DIR) -> list[Page]:
    """Load every PDF in docs_dir, in a stable (sorted) order."""
    paths = sorted(docs_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")
    return [page for path in paths for page in load_pdf(path)]
