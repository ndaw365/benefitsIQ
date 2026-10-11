"""The golden set is data, so test it like code: every quote must exist on its cited page."""

import pytest
from pydantic import ValidationError

from src.eval.golden import load_golden
from src.rag.load import load_all
from src.schemas import GoldenItem

ITEMS = load_golden()
PAGES = {(p.doc_id, p.page): p.text for p in load_all()}


@pytest.mark.parametrize("item", ITEMS, ids=[i.id for i in ITEMS])
def test_quotes_appear_on_cited_page(item: GoldenItem):
    for source in item.sources:
        page_text = PAGES.get((source.doc_id, source.page))
        assert page_text is not None, f"{source.doc_id} p{source.page} not in the loaded docs"
        assert source.quote in page_text, f"quote not found on {source.doc_id} p{source.page}"


def test_size_and_mix():
    categories = {item.category for item in ITEMS}
    assert 15 <= len(ITEMS) <= 40
    assert {"fact", "table", "multi_policy", "paraphrase", "unanswerable"} <= categories


def test_unanswerable_with_sources_rejected():
    with pytest.raises(ValidationError):
        GoldenItem(id="x", question="q", expected_answer="a", category="unanswerable",
                   sources=[{"doc_id": "d", "page": 1, "quote": "some quote text"}])
