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


def test_contextual_collection_name():
    assert collection_name(800, 150, contextual=True) == "policies_s800_o150_ctx"


def test_contextual_header_is_embedded_but_not_stored():
    pages = [Page("lincoln_moravian_ltd", 28, "due to intentionally self-inflicted injuries " * 3)]
    chunks = chunk_pages(pages, size=300, overlap=50)
    seen: list[str] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        seen.extend(texts)
        return fake_embed(texts)

    collection = build_index(chunks, recording_embed, chromadb.EphemeralClient(), "test_ctx", contextual=True)
    assert seen[0].startswith("Lincoln Financial Group") and seen[0].endswith(chunks[0].text)
    stored = collection.get(ids=[chunks[0].chunk_id], include=["documents"])["documents"][0]
    assert stored == chunks[0].text  # prompts and citation spans see the raw text


def test_unknown_doc_fails_loudly_in_contextual_mode():
    import pytest
    chunks = chunk_pages([Page("mystery_doc", 1, "text " * 20)], size=300, overlap=50)
    with pytest.raises(KeyError, match="DOC_TITLES"):
        build_index(chunks, fake_embed, chromadb.EphemeralClient(), "test_unknown", contextual=True)
