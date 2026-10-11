"""Agent loop and guardrails with a scripted fake model: no API calls."""

import uuid

import chromadb
import pytest

from src.agent.loop import check_answer, run_agent, Step
from src.agent.tools import AgentTools
from src.llm import ToolCall, ToolTurn

PASSAGE_ID = "lincoln_moravian_ltd:p3:c0"
SEARCH = ToolCall("search_policy", {"query": "Moravian benefit percentage and maximum"})
CALC = ToolCall("calculate_benefit", {"monthly_earnings": 5400, "benefit_percent": 60, "max_monthly_benefit": 10000})


class ScriptedChat:
    """Plays back a fixed list of model turns, recording what the loop sent."""

    def __init__(self, turns: list[ToolTurn]):
        self.turns, self.sent = list(turns), []

    def __call__(self, system, specs):  # acts as the chat_factory
        return self

    def send(self, text):
        self.sent.append(text)
        return self.turns.pop(0)

    def send_tool_results(self, results):
        self.sent.append(results)
        return self.turns.pop(0)


@pytest.fixture
def tools(tmp_path) -> AgentTools:
    collection = chromadb.EphemeralClient().create_collection(
        f"t_loop_{uuid.uuid4().hex[:8]}", configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=[PASSAGE_ID], embeddings=[[1.0, 0.0]],
                   documents=["BENEFIT PERCENTAGE: 60% MAXIMUM MONTHLY BENEFIT: $10,000"],
                   metadatas=[{"doc_id": "lincoln_moravian_ltd", "page": 3, "start_char": 0}])
    return AgentTools(lambda texts: [[1.0, 0.0] for _ in texts], collection, escalation_log=tmp_path / "e.jsonl")


def test_search_then_calculate_then_cited_answer(tools):
    chat = ScriptedChat([ToolTurn(None, [SEARCH]), ToolTurn(None, [CALC]),
                         ToolTurn(f"Estimated benefit is $3,240 per month: 60% up to $10,000 [{PASSAGE_ID}].", [])])
    result = run_agent("Estimate the benefit", tools, chat_factory=chat)
    assert [s.tool for s in result.steps] == ["search_policy", "calculate_benefit"]
    assert result.steps[1].output["net"] == 3240.0
    assert result.cited_ids == [PASSAGE_ID] and not result.needs_review


def test_escalation_stops_the_loop(tools):
    escalate = ToolCall("escalate_to_human", {"reason": "claim decision requested", "summary": "claim_02"})
    chat = ScriptedChat([ToolTurn(None, [escalate]), ToolTurn("should never be reached", [])])
    result = run_agent("Approve claim_02?", tools, chat_factory=chat)
    assert result.escalated and result.ticket.startswith("ESC-") and len(chat.turns) == 1


def test_step_limit_escalates(tools):
    chat = ScriptedChat([ToolTurn(None, [SEARCH])] * 10)
    result = run_agent("loop forever", tools, chat_factory=chat, max_steps=3)
    assert result.escalated and "step limit" in result.answer and len(result.steps) == 3


def test_tool_error_goes_back_to_model(tools):
    bad = ToolCall("calculate_benefit", {"monthly_earnings": 5400})
    chat = ScriptedChat([ToolTurn(None, [bad]), ToolTurn("The documents do not say.", [])])
    result = run_agent("q", tools, chat_factory=chat)
    assert "error" in result.steps[0].output and "error" in chat.sent[1][0][1]


def test_guardrail_invented_citation():
    _, invalid, reasons = check_answer("60% [standard_uah_ltd:p99:c0].", [], seen_ids={PASSAGE_ID})
    assert invalid == ["standard_uah_ltd:p99:c0"] and any("never retrieved" in r for r in reasons)


def test_guardrail_no_evidence():
    _, _, reasons = check_answer("The benefit is 60%.", [], seen_ids=set())
    assert reasons == ["no cited evidence for the answer"]


def test_guardrail_calculate_before_search():
    steps = [Step("calculate_benefit", {}, {}), Step("search_policy", {}, {})]
    _, _, reasons = check_answer(f"$3,240 [{PASSAGE_ID}]", steps, seen_ids={PASSAGE_ID})
    assert reasons == ["benefit calculated before searching the policy for its inputs"]


def test_guardrail_decision_language():
    _, _, reasons = check_answer(f"The claim is approved [{PASSAGE_ID}].", [], seen_ids={PASSAGE_ID})
    assert reasons == ["answer uses claim-decision language"]


def test_guardrail_claim_review_issues_propagate():
    steps = [Step("get_claim", {"claim_id": "claim_05"}, {"claim": {}, "review_issues": ["24.0 hours/week < 30"]})]
    _, _, reasons = check_answer(f"Estimated $186 [{PASSAGE_ID}].", steps, seen_ids={PASSAGE_ID})
    assert reasons == ["claim claim_05 has review issues: ['24.0 hours/week < 30']"]
