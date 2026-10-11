"""Retrieve the chunks nearest to a question from a Chroma collection."""

import argparse
from dataclasses import dataclass

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from src.rag.bm25 import BM25
from src.rag.index import CHROMA_DIR, DEFAULT_OVERLAP, DEFAULT_SIZE, EmbedFn, collection_name, make_embedder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    start_char: int  # offset of the chunk within its page text
    text: str
    distance: float  # cosine distance = 1 - similarity; lower is closer


def retrieve(
    question: str, embed: EmbedFn, collection: Collection, k: int = 5, doc_ids: list[str] | None = None
) -> list[Hit]:
    """Return up to k chunks closest to the question, nearest first. Empty list if nothing is indexed.

    doc_ids restricts the search to those documents; None or [] searches everything.
    """
    if not question.strip():
        raise ValueError("question is empty")
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=embed([question]),  # must be the same model that embedded the chunks
        n_results=min(k, collection.count()),
        where={"doc_id": {"$in": doc_ids}} if doc_ids else None,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns one list per query; we sent one query, so take index [0] of each.
    return [
        Hit(chunk_id, meta["doc_id"], meta["page"], meta["start_char"], text, distance)
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


RRF_K = 60  # standard constant (Cormack et al., 2009); larger values flatten the gap between ranks


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) per id. Uses ranks only, never raw scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def build_bm25(collection: Collection) -> BM25:
    """Keyword index over exactly the chunks in the collection, so both retrievers see the same corpus."""
    data = collection.get(include=["documents"])
    return BM25(data["ids"], data["documents"])


def _cosine_distance(a: list[float], b: list[float]) -> float:
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def hybrid_retrieve(
    question: str, embed: EmbedFn, collection: Collection, bm25: BM25, k: int = 5, candidates: int = 20
) -> list[Hit]:
    """Fuse the top `candidates` from vector search and BM25 with RRF; return the top k as Hits."""
    vector_hits = retrieve(question, embed, collection, candidates)  # also validates the question
    if not vector_hits:
        return []
    keyword_ids = [chunk_id for chunk_id, _ in bm25.search(question, candidates)]
    fused = rrf([[h.chunk_id for h in vector_hits], keyword_ids])[:k]

    hits = {h.chunk_id: h for h in vector_hits}
    keyword_only = [chunk_id for chunk_id in fused if chunk_id not in hits]
    if keyword_only:  # fetch text and metadata, and compute the vector distance so it stays comparable
        q_vec = embed([question])[0]
        data = collection.get(ids=keyword_only, include=["documents", "metadatas", "embeddings"])
        for chunk_id, text, meta, emb in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
            hits[chunk_id] = Hit(chunk_id, meta["doc_id"], meta["page"], meta["start_char"], text,
                                 _cosine_distance(q_vec, emb))
    return [hits[chunk_id] for chunk_id in fused]


def open_collection(size: int, overlap: int) -> Collection:
    """Open the persisted collection for a chunking config, with a clear error if it isn't built."""
    name = collection_name(size, overlap)
    try:
        return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(name)
    except NotFoundError:
        raise SystemExit(f"No collection {name!r}. Build it: python -m src.rag.index --size {size} --overlap {overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the top-k chunks for a question (no LLM).")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    hits = retrieve(args.question, make_embedder(), open_collection(args.size, args.overlap), args.k)
    if not hits:
        print("No chunks retrieved.")
    for rank, hit in enumerate(hits, start=1):
        print(f"\n#{rank}  {hit.doc_id} p.{hit.page}  distance={hit.distance:.3f}")
        print(f"    {hit.text[:300]}...")


if __name__ == "__main__":
    main()
