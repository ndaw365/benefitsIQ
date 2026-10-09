"""Index-building tests with a fake embedder and an in-memory Chroma client (no model download)."""

import chromadb

from src.rag.chunk import chunk_pages
from src.rag.index import build_index, collection_name
from src.rag.load import Page

PAGES = [Page("doc_a", 1, "alpha " * 100), Page("doc_b", 3, "beta " * 100)]


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_collection_name_encodes_config():
    assert collection_name(800, 150) == "policies_s800_o150"


def test_build_index_stores_every_chunk_with_metadata():
    chunks = chunk_pages(PAGES, size=300, overlap=50)
    client = chromadb.EphemeralClient()
    collection = build_index(chunks, fake_embed, client, "test_store")

    assert collection.count() == len(chunks)
    stored = collection.get(ids=[chunks[0].chunk_id], include=["documents", "metadatas"])
    assert stored["documents"][0] == chunks[0].text
    assert stored["metadatas"][0] == {"doc_id": "doc_a", "page": 1, "start_char": 0}


def test_rebuild_does_not_duplicate():
    chunks = chunk_pages(PAGES, size=300, overlap=50)
    client = chromadb.EphemeralClient()
    build_index(chunks, fake_embed, client, "test_rebuild")
    collection = build_index(chunks, fake_embed, client, "test_rebuild")
    assert collection.count() == len(chunks)
