"""Retrieval hit rate@k on the golden set, per chunking config. No LLM calls."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from src.eval.golden import load_golden
from src.rag.index import make_embedder
from src.rag.load import load_all
from src.rag.retrieve import Hit, build_bm25, hybrid_retrieve, open_collection, retrieve
from src.rag.route import detect_policies, strip_policy_names
from src.schemas import GoldenItem, GoldenSource

RESULTS_DIR = Path("results/retrieval")
PageTexts = dict[tuple[str, int], str]


@dataclass(frozen=True)
class ItemResult:
    id: str
    category: str
    page_hit: bool  # some top-k chunk is on a labeled page (lenient)
    span_hit: bool  # some top-k chunk overlaps a labeled quote (strict)
    first_page_rank: int | None  # 1-based rank of the first page hit; None if no hit
    top1_distance: float


def overlaps_quote(hit: Hit, source: GoldenSource, pages: PageTexts) -> bool:
    """True if the chunk is on the source page and its character span overlaps the quote's span."""
    if (hit.doc_id, hit.page) != (source.doc_id, source.page):
        return False
    q_start = pages[(source.doc_id, source.page)].index(source.quote)  # test_golden guarantees it exists
    q_end = q_start + len(source.quote)
    return hit.start_char < q_end and q_start < hit.start_char + len(hit.text)


def score_item(item: GoldenItem, hits: list[Hit], pages: PageTexts) -> ItemResult:
    """Score one golden item against its retrieved hits (ranked nearest first).

    A hit requires every cited document to be covered: for a multi-policy question,
    finding only one policy's answer is a miss. Within a document, any source counts.
    """
    labeled = {(s.doc_id, s.page) for s in item.sources}
    page_ranks = [rank for rank, h in enumerate(hits, start=1) if (h.doc_id, h.page) in labeled]
    docs = {s.doc_id for s in item.sources}
    page_hit = all(any((h.doc_id, h.page) in labeled for h in hits if h.doc_id == d) for d in docs)
    span_hit = all(
        any(overlaps_quote(h, s, pages) for h in hits for s in item.sources if s.doc_id == d) for d in docs
    )
    return ItemResult(
        id=item.id,
        category=item.category,
        page_hit=page_hit,
        span_hit=span_hit,
        first_page_rank=page_ranks[0] if page_ranks else None,
        top1_distance=hits[0].distance if hits else float("nan"),
    )


def summarize(results: list[ItemResult]) -> dict:
    """Hit rates over answerable items only; top-1 distance split by answerable vs not."""
    answerable = [r for r in results if r.category != "unanswerable"]
    unanswerable = [r for r in results if r.category == "unanswerable"]
    by_category = {
        cat: round(mean(r.span_hit for r in answerable if r.category == cat), 3)
        for cat in sorted({r.category for r in answerable})
    }
    return {
        "n_answerable": len(answerable),
        "page_hit_rate": round(mean(r.page_hit for r in answerable), 3),
        "span_hit_rate": round(mean(r.span_hit for r in answerable), 3),
        "span_hit_rate_by_category": by_category,
        "mean_top1_distance_answerable": round(mean(r.top1_distance for r in answerable), 3),
        "mean_top1_distance_unanswerable": round(mean(r.top1_distance for r in unanswerable), 3) if unanswerable else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hit rate@k on the golden set.")
    parser.add_argument("--configs", nargs="+", default=["800:150", "400:50"], help="size:overlap pairs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--route", action="store_true", help="restrict search to policies named in the question")
    parser.add_argument("--strip", action="store_true", help="with --route, remove policy names from the query text")
    args = parser.parse_args()
    if args.route and args.mode == "hybrid":
        parser.error("--route is only implemented for --mode vector")

    items, pages, embed = load_golden(), {(p.doc_id, p.page): p.text for p in load_all()}, make_embedder()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        size, overlap = map(int, config.split(":"))
        collection = open_collection(size, overlap)
        bm25 = build_bm25(collection) if args.mode == "hybrid" else None

        def search(question: str) -> list[Hit]:
            if bm25 is None:
                doc_ids = detect_policies(question) if args.route else None
                query = strip_policy_names(question) if doc_ids and args.strip else question
                return retrieve(query, embed, collection, args.k, doc_ids)
            return hybrid_retrieve(question, embed, collection, bm25, args.k)

        results = [score_item(item, search(item.question), pages) for item in items]
        summary = summarize(results)
        if args.route:  # a routing error excludes a document the answer needs; filtering can't recover from that
            summary["routing_errors"] = [
                item.id for item in items
                if (routed := detect_policies(item.question)) and not {s.doc_id for s in item.sources} <= set(routed)
            ]

        run = args.mode + ("_routed" if args.route else "") + ("_stripped" if args.route and args.strip else "")
        print(f"\n=== {collection.name}  {run}  k={args.k}")
        for r in results:
            mark = "SPAN" if r.span_hit else ("page" if r.page_hit else ("  - " if r.category == "unanswerable" else "MISS"))
            print(f"  {mark:4}  rank={str(r.first_page_rank or '-'):>2}  d1={r.top1_distance:.3f}  {r.id}")
        print(json.dumps(summary, indent=2))

        out = RESULTS_DIR / f"{collection.name}_{run}_k{args.k}.json"
        out.write_text(json.dumps({"summary": summary, "items": [asdict(r) for r in results]}, indent=2))


if __name__ == "__main__":
    main()
