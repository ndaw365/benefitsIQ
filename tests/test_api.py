"""API tests with FastAPI's TestClient and dependency overrides: real HTTP handling, fake models."""

import json
import uuid
from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.llm import EmptyResponseError, ToolCall, ToolTurn
from src.schemas import ClaimExtraction, PolicyAnswer

PASSAGE = "lincoln_moravian_ltd:p3:c1"
LABELS = [json.loads(line) for line in Path("data/claims/labels.jsonl").read_text().splitlines()]


def fake_llm(prompt, *, schema, **kwargs):
    """Answers RAG questions and extracts claim_05, depending on the requested schema."""
    if schema is PolicyAnswer:
        return PolicyAnswer(answer="180 calendar days.", confidence="high", sources=[PASSAGE], needs_human_review=False)
    return ClaimExtraction.model_validate(LABELS[4]["expected"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    collection = chromadb.EphemeralClient().create_collection(
        f"t_api_{uuid.uuid4().hex[:8]}", configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=[PASSAGE], embeddings=[[1.0, 0.0]], documents=["ELIMINATION PERIOD: 180 calendar days"],
                   metadatas=[{"doc_id": "lincoln_moravian_ltd", "page": 3, "start_char": 650}])
    monkeypatch.setattr("src.agent.loop.RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr("src.agent.tools.ESCALATION_LOG", tmp_path / "esc.jsonl")
    main.app.dependency_overrides = {
        main.get_resources: lambda: (lambda texts: [[1.0, 0.0] for _ in texts], collection),
        main.get_llm: lambda: fake_llm,
    }
    yield TestClient(main.app)
    main.app.dependency_overrides = {}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_with_readable_sources(client):
    body = client.post("/ask", json={"question": "What is the elimination period at Moravian?"}).json()
    assert body["answer"]["answer"] == "180 calendar days."
    assert body["sources"][0]["page"] == 3 and "Moravian University" in body["sources"][0]["label"]
    assert body["needs_review"] is False


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 1001}, {}, {"q": "wrong field"}])
def test_invalid_requests_rejected_before_any_model_call(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_extract_flags_planted_issue(client):
    form = Path("data/claims/claim_05.txt").read_text()
    body = client.post("/claims/extract", json={"form_text": form}).json()
    assert body["claim"]["hours_per_week"] == 24 and body["needs_review"] is True
    assert [i["code"] for i in body["issues"]] == ["below_min_hours"]


def test_model_failure_maps_to_502(client):
    def broken_llm(prompt, **kwargs):
        raise EmptyResponseError("blocked")
    main.app.dependency_overrides[main.get_llm] = lambda: broken_llm
    response = client.post("/ask", json={"question": "What is the elimination period?"})
    assert response.status_code == 502 and "unusable output" in response.json()["detail"]


def test_agent_endpoint_runs_loop_and_reports_escalation(client):
    escalate = ToolCall("escalate_to_human", {"reason": "claim decision requested", "summary": "claim_02"})

    class OneTurnChat:
        def __init__(self, system, specs): pass
        def send(self, text): return ToolTurn(None, [escalate])

    main.app.dependency_overrides[main.get_chat_factory] = lambda: OneTurnChat
    body = client.post("/agent", json={"question": "Approve claim_02?"}).json()
    assert body["escalated"] is True and body["needs_review"] is True
    assert body["review_reasons"][0].startswith("escalated") and body["steps"][0]["tool"] == "escalate_to_human"
