"""Embed chunks with a local sentence-transformers model and store them in Chroma."""

import argparse
import time
from collections.abc import Callable

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.rag.chunk import Chunk, chunk_pages
from src.rag.load import DOC_TITLES, load_all

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, max 256 tokens per input
CHROMA_DIR = ".chroma"
DEFAULT_SIZE, DEFAULT_OVERLAP = 800, 150  # characters; ~800 chars stays under the 256-token limit

EmbedFn = Callable[[list[str]], list[list[float]]]


def make_embedder(model_name: str = EMBED_MODEL) -> EmbedFn:
    """Load the model once and return a function that embeds a list of texts."""
    from sentence_transformers import SentenceTransformer  # slow import; only when needed

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        # Unit-length vectors make cosine similarity a plain dot product.
        return model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()

    return embed


def collection_name(size: int, overlap: int, contextual: bool = False) -> str:
    """One collection per chunking config, so we can compare configs side by side in evals."""
    return f"policies_s{size}_o{overlap}" + ("_ctx" if contextual else "")


def embedding_text(chunk: Chunk, contextual: bool) -> str:
    """Text the embedder sees. Contextual mode prefixes the document title, so every chunk
    carries its policy's identity, not just the cover and schedule pages."""
    if not contextual:
        return chunk.text
    if chunk.doc_id not in DOC_TITLES:
        raise KeyError(f"No DOC_TITLES entry for {chunk.doc_id!r}; add one in src/rag/load.py")
    return f"{DOC_TITLES[chunk.doc_id]}\n{chunk.text}"


def build_index(
    chunks: list[Chunk], embed: EmbedFn, client: ClientAPI, name: str, contextual: bool = False
) -> Collection:
    """(Re)create collection `name` holding every chunk, its vector, and its citation metadata.

    Only the embedding sees the header; the stored document stays the raw chunk text,
    so prompts, citations, and golden quote spans are unchanged.
    """
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)  # rebuild from scratch: no stale or duplicate chunks
    collection = client.create_collection(name, configuration={"hnsw": {"space": "cosine"}})

    batch = client.get_max_batch_size()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=embed([embedding_text(c, contextual) for c in part]),
            documents=[c.text for c in part],
            metadatas=[{"doc_id": c.doc_id, "page": c.page, "start_char": c.start_char} for c in part],
        )
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma index for one chunking config.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--contextual", action="store_true", help="prefix each chunk's document title before embedding")
    args = parser.parse_args()

    chunks = chunk_pages(load_all(), args.size, args.overlap)
    start = time.perf_counter()
    collection = build_index(
        chunks, make_embedder(), chromadb.PersistentClient(path=CHROMA_DIR),
        collection_name(args.size, args.overlap, args.contextual), args.contextual,
    )
    print(f"{collection.name}: {collection.count()} chunks indexed in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
