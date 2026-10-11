"""BM25 ranking and reciprocal rank fusion on tiny hand-made corpora."""

import chromadb

from src.rag.bm25 import BM25, tokenize
from src.rag.retrieve import build_bm25, hybrid_retrieve, rrf

TEXTS = {
    "exclusions": "The policy will not cover disability due to intentionally self-inflicted injuries.",
    "cover": "Certificate of group long-term disability insurance. Lincoln policy certificate.",
    "benefit": "The policy pays a monthly benefit after the elimination period. The policy is group policy.",
}
INDEX = BM25(list(TEXTS), list(TEXTS.values()))


def test_tokenize_lowercases_splits_hyphens_and_drops_stopwords():
    assert tokenize("Is a Self-Inflicted injury covered?") == ["self", "inflicted", "injury", "covered"]


def test_rare_term_beats_common_term():
    top_id, _ = INDEX.search("self-inflicted policy")[0]
    assert top_id == "exclusions"  # 'inflicted' is in 1 chunk, 'policy' in 3


def test_rare_terms_have_higher_idf():
    assert INDEX.idf["inflicted"] > INDEX.idf["policy"]


def test_no_overlap_returns_nothing():
    assert INDEX.search("dental cleanings") == []


def test_repeated_term_saturates():
    one = BM25(["a"], ["benefit"]).search("benefit")[0][1]
    many = BM25(["a"], ["benefit " * 20]).search("benefit")[0][1]
    assert many < 3 * one  # 20x the occurrences is nowhere near 20x the score


def test_rrf_rewards_agreement():
    # "both" is 2nd in each list; "only_a" is 1st in one list only. Agreement wins: 2/62 > 1/61.
    assert rrf([["only_a", "both"], ["only_b", "both"]])[0] == "both"


def test_hybrid_returns_keyword_only_hit_with_distance():
    collection = chromadb.EphemeralClient().create_collection("t_hybrid", configuration={"hnsw": {"space": "cosine"}})
    vectors = {"exclusions": [0.0, 1.0], "cover": [1.0, 0.0], "benefit": [0.9, 0.1]}
    collection.add(ids=list(TEXTS), embeddings=list(vectors.values()), documents=list(TEXTS.values()),
                   metadatas=[{"doc_id": "doc", "page": i, "start_char": 0} for i in range(len(TEXTS))])

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]  # every query looks like the cover page

    hits = hybrid_retrieve("self-inflicted", embed, collection, build_bm25(collection), k=3, candidates=1)
    ids = [h.chunk_id for h in hits]
    assert ids[:2] == ["cover", "exclusions"] or ids[:2] == ["exclusions", "cover"]
    exclusions = next(h for h in hits if h.chunk_id == "exclusions")
    assert abs(exclusions.distance - 1.0) < 1e-6  # orthogonal to the query vector
