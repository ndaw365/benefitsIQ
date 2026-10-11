"""Policy routing: which documents a question names, and filtered retrieval."""

import chromadb

from src.rag.retrieve import retrieve
from src.rag.route import detect_policies


def test_single_policy_by_alias():
    assert detect_policies("What is the elimination period at Moravian?") == ["lincoln_moravian_ltd"]
    assert detect_policies("Benefit waiting period under the UAH plan?") == ["standard_uah_ltd"]


def test_no_policy_named_searches_everything():
    assert detect_policies("How long is the own occupation period?") == []


def test_generic_word_standard_does_not_route():
    assert detect_policies("Is this a standard exclusion?") == []


def test_whole_words_only():
    assert detect_policies("Does lincolnshire count?") == []


def test_two_policies():
    assert set(detect_policies("Compare Lincoln and Huntsville elimination periods")) == {
        "lincoln_moravian_ltd", "standard_uah_ltd"}


def test_retrieve_respects_doc_filter():
    collection = chromadb.EphemeralClient().create_collection("t_route", configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=["a:p1:c0", "b:p1:c0"], embeddings=[[1.0, 0.0], [0.0, 1.0]], documents=["near", "far"],
                   metadatas=[{"doc_id": "a", "page": 1, "start_char": 0}, {"doc_id": "b", "page": 1, "start_char": 0}])

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]  # nearest is doc a

    assert [h.doc_id for h in retrieve("q", embed, collection, k=2, doc_ids=["b"])] == ["b"]
    assert [h.doc_id for h in retrieve("q", embed, collection, k=2)] == ["a", "b"]
