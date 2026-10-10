"""Answer-step tests with a fake LLM: no API calls, deterministic."""

import chromadb
import pytest

from src.rag.answer import NOT_FOUND, answer_question, build_prompt, check_citations
from src.rag.retrieve import Hit
from src.schemas import PolicyAnswer

HITS = [
    Hit("doc_a:p3:c0", "doc_a", 3, 0, "Elimination period: 180 days.", 0.2),
    Hit("doc_b:p8:c1", "doc_b", 8, 650, "Benefit waiting period: 90 days.", 0.3),
]


def make_answer(sources: list[str]) -> PolicyAnswer:
    return PolicyAnswer(answer="180 days.", confidence="high", sources=sources, needs_human_review=False)


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


def test_prompt_contains_ids_pages_and_question():
    prompt = build_prompt("How long?", HITS)
    assert "[doc_a:p3:c0] (doc_a, page 3)" in prompt
    assert "Benefit waiting period: 90 days." in prompt
    assert prompt.endswith("Question: How long?")


def test_valid_citations_pass_unchanged():
    answer, invalid = check_citations(make_answer(["doc_a:p3:c0"]), HITS)
    assert invalid == [] and answer.sources == ["doc_a:p3:c0"] and not answer.needs_human_review


def test_invented_citation_is_removed_and_flagged():
    answer, invalid = check_citations(make_answer(["doc_a:p3:c0", "doc_z:p99:c0"]), HITS)
    assert invalid == ["doc_z:p99:c0"]
    assert answer.sources == ["doc_a:p3:c0"]
    assert answer.needs_human_review


def test_empty_collection_skips_llm():
    def llm_must_not_run(*args, **kwargs):
        pytest.fail("LLM called with no retrieved context")

    collection = chromadb.EphemeralClient().create_collection("t_answer_empty")
    result = answer_question("How long?", fake_embed, collection, llm=llm_must_not_run)
    assert result.answer == NOT_FOUND and result.hits == []


def test_llm_receives_schema_and_hits():
    collection = chromadb.EphemeralClient().create_collection("t_answer_hits")
    collection.add(ids=["doc_a:p3:c0"], embeddings=[[1.0, 0.0]], documents=["Elimination period: 180 days."],
                   metadatas=[{"doc_id": "doc_a", "page": 3, "start_char": 0}])
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return make_answer(["doc_a:p3:c0"])

    result = answer_question("How long?", fake_embed, collection, llm=fake_llm)
    assert calls[0][1]["schema"] is PolicyAnswer
    assert "[doc_a:p3:c0]" in calls[0][0]
    assert result.answer.sources == ["doc_a:p3:c0"] and result.invalid_sources == []
