"""Tool execution: dispatch, error handling, and the state guardrails rely on."""

import json
import uuid

import chromadb
import pytest

from src.agent.tools import AgentTools
from src.llm import ToolCall


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def tools(tmp_path) -> AgentTools:
    name = f"t_agent_{uuid.uuid4().hex[:8]}"  # unique: in-memory clients share state across tests
    collection = chromadb.EphemeralClient().create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=["lincoln_moravian_ltd:p3:c0"], embeddings=[[1.0, 0.0]],
                   documents=["BENEFIT PERCENTAGE: 60% MAXIMUM MONTHLY BENEFIT: $10,000"],
                   metadatas=[{"doc_id": "lincoln_moravian_ltd", "page": 3, "start_char": 0}])
    return AgentTools(fake_embed, collection, escalation_log=tmp_path / "esc.jsonl")


def test_search_returns_citable_passages_and_records_ids(tools):
    out = tools.run(ToolCall("search_policy", {"query": "Moravian benefit percentage"}))
    assert out["passages"][0]["id"] == "lincoln_moravian_ltd:p3:c0"
    assert tools.seen_ids == {"lincoln_moravian_ltd:p3:c0"}


def test_calculate_benefit_returns_steps(tools):
    out = tools.run(ToolCall("calculate_benefit", {"monthly_earnings": 5400, "benefit_percent": 60, "max_monthly_benefit": 10000}))
    assert out["net"] == 3240.0 and len(out["steps"]) == 4


def test_bad_arguments_come_back_as_errors(tools):
    assert "error" in tools.run(ToolCall("calculate_benefit", {"monthly_earnings": 5400}))  # missing args
    assert "error" in tools.run(ToolCall("calculate_benefit", {"monthly_earnings": -1, "benefit_percent": 60,
                                                               "max_monthly_benefit": 10000}))  # impossible value


def test_unknown_tool_is_an_error_not_a_crash(tools):
    assert "Unknown tool" in tools.run(ToolCall("approve_claim", {}))["error"]


def test_escalation_is_logged(tools, tmp_path):
    out = tools.run(ToolCall("escalate_to_human", {"reason": "missing earnings", "summary": "none"}))
    assert out["status"] == "escalated" and tools.escalation["ticket"] == out["ticket"]
    assert json.loads((tmp_path / "esc.jsonl").read_text())["reason"] == "missing earnings"


def test_get_claim_rejects_path_tricks(tools):
    for bad in ["../../.env", "claim_05.txt", "claim_5", "CLAIM_05"]:
        assert "error" in tools.run(ToolCall("get_claim", {"claim_id": bad}))


def test_get_claim_runs_extraction_and_review(tmp_path):
    from src.extraction.extract import ClaimResult
    from src.extraction.review import Issue
    from src.schemas import ClaimExtraction
    labels = [json.loads(line) for line in open("data/claims/labels.jsonl")]
    claim = ClaimExtraction.model_validate(labels[4]["expected"])  # claim_05
    fake_extract = lambda text: ClaimResult(claim, [Issue("below_min_hours", "24.0 hours/week < 30")])
    tools = AgentTools(fake_embed, None, escalation_log=tmp_path / "e.jsonl", extract=fake_extract)
    out = tools.run(ToolCall("get_claim", {"claim_id": "claim_05"}))
    assert out["claim"]["monthly_earnings"] == 3100 and out["review_issues"] == ["24.0 hours/week < 30"]
