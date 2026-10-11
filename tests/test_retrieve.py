"""Retrieval tests with hand-made 2-D vectors, so the expected nearest chunk is obvious."""

import chromadb
import pytest

from src.rag.retrieve import retrieve

VECTORS = {"cat": [1.0, 0.0], "dog": [0.9, 0.1], "tax": [0.0, 1.0]}


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [VECTORS[t] for t in texts]


def make_collection(name: str, words: list[str]):
    collection = chromadb.EphemeralClient().create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    if words:
        collection.add(
            ids=[f"doc:p{i}:c0" for i in range(len(words))],
            embeddings=fake_embed(words),
            documents=words,
            metadatas=[{"doc_id": "doc", "page": i, "start_char": 0} for i in range(len(words))],
        )
    return collection


def test_nearest_first():
    hits = retrieve("cat", fake_embed, make_collection("t_order", ["tax", "dog", "cat"]), k=3)
    assert [h.text for h in hits] == ["cat", "dog", "tax"]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[0].distance < hits[1].distance < hits[2].distance


def test_k_larger_than_collection():
    assert len(retrieve("cat", fake_embed, make_collection("t_small", ["dog"]), k=5)) == 1


def test_empty_collection_returns_empty_list():
    assert retrieve("cat", fake_embed, make_collection("t_empty", []), k=5) == []


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        retrieve("   ", fake_embed, make_collection("t_blank", ["cat"]))
