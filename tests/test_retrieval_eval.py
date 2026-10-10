"""Scoring logic for the retrieval eval, on a tiny hand-made page."""

from src.eval.retrieval import score_item, summarize
from src.rag.retrieve import Hit
from src.schemas import GoldenItem

PAGE = "Intro text here. ELIMINATION PERIOD: 180 calendar days. Other schedule rows follow."
PAGES = {("doc", 3): PAGE}
QUOTE_START = PAGE.index("ELIMINATION")
ITEM = GoldenItem(id="elim", question="q", expected_answer="180", category="fact",
                  sources=[{"doc_id": "doc", "page": 3, "quote": "ELIMINATION PERIOD: 180 calendar days"}])


def hit(page: int, start: int, text: str, distance: float = 0.3) -> Hit:
    return Hit(f"doc:p{page}:c{start}", "doc", page, start, text, distance)


def test_span_hit_when_chunk_covers_quote():
    result = score_item(ITEM, [hit(3, QUOTE_START, PAGE[QUOTE_START:])], PAGES)
    assert result.page_hit and result.span_hit and result.first_page_rank == 1


def test_page_hit_without_span_when_chunk_misses_quote():
    result = score_item(ITEM, [hit(3, 0, PAGE[:15])], PAGES)  # "Intro text here" only
    assert result.page_hit and not result.span_hit


def test_partial_overlap_counts_as_span_hit():
    result = score_item(ITEM, [hit(3, 0, PAGE[: QUOTE_START + 5])], PAGES)
    assert result.span_hit


def test_miss_and_rank():
    result = score_item(ITEM, [hit(9, 0, "unrelated"), hit(3, 0, PAGE)], PAGES)
    assert result.first_page_rank == 2
    assert not score_item(ITEM, [hit(9, 0, "unrelated")], PAGES).page_hit


def test_summary_excludes_unanswerable_from_hit_rate():
    unanswerable = GoldenItem(id="u", question="q", expected_answer="none", category="unanswerable", sources=[])
    results = [score_item(ITEM, [hit(3, QUOTE_START, PAGE[QUOTE_START:], 0.2)], PAGES),
               score_item(unanswerable, [hit(9, 0, "x", 0.6)], PAGES)]
    summary = summarize(results)
    assert summary["n_answerable"] == 1 and summary["span_hit_rate"] == 1.0
    assert summary["mean_top1_distance_unanswerable"] == 0.6


def test_multi_policy_needs_every_document():
    pages = {("doc", 3): PAGE, ("other", 8): "Benefit Waiting Period: The later of: 90 days."}
    item = GoldenItem(id="multi", question="q", expected_answer="a", category="multi_policy", sources=[
        {"doc_id": "doc", "page": 3, "quote": "ELIMINATION PERIOD: 180 calendar days"},
        {"doc_id": "other", "page": 8, "quote": "Benefit Waiting Period"}])
    one = hit(3, QUOTE_START, PAGE[QUOTE_START:])
    both = Hit("other:p8:c0", "other", 8, 0, pages[("other", 8)], 0.3)
    assert not score_item(item, [one], pages).span_hit
    assert not score_item(item, [one], pages).page_hit
    assert score_item(item, [one, both], pages).span_hit
