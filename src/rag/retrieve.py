"""Retrieve the chunks nearest to a question from a Chroma collection."""

import argparse
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from src.rag.index import CHROMA_DIR, DEFAULT_OVERLAP, DEFAULT_SIZE, EmbedFn, collection_name, make_embedder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    distance: float  # cosine distance = 1 - similarity; lower is closer


def retrieve(question: str, embed: EmbedFn, collection: Collection, k: int = 5) -> list[Hit]:
    """Return up to k chunks closest to the question, nearest first. Empty list if nothing is indexed."""
    if not question.strip():
        raise ValueError("question is empty")
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=embed([question]),  # must be the same model that embedded the chunks
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns one list per query; we sent one query, so take index [0] of each.
    return [
        Hit(chunk_id, meta["doc_id"], meta["page"], text, distance)
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


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
