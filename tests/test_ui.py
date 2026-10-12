"""UI helpers, plus a smoke test that the Streamlit script renders without errors or API calls."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.agent.loop import AgentResult
from src.rag.answer import RagResult
from src.rag.retrieve import Hit
from src.schemas import PolicyAnswer
from src.views import agent_review_reasons, cited_sources, qa_review_reasons, source_label

APP = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"  # AppTest resolves relative to this file
HIT = Hit("lincoln_moravian_ltd:p3:c1", "lincoln_moravian_ltd", 3, 650, "ELIMINATION PERIOD: 180 calendar days", 0.3)


def answer(**overrides) -> PolicyAnswer:
    base = dict(answer="180 calendar days.", confidence="high", sources=[HIT.chunk_id], needs_human_review=False)
    return PolicyAnswer(**{**base, **overrides})


def test_source_label_uses_document_title():
    assert source_label("lincoln_moravian_ltd", 3).endswith("Moravian University, page 3")


def test_clean_answer_has_no_reasons_and_readable_sources():
    result = RagResult(answer(), [HIT], [])
    assert qa_review_reasons(result) == [] and len(cited_sources(result)) == 1


def test_review_reasons_explain_each_flag():
    result = RagResult(answer(sources=[], needs_human_review=True, confidence="low"), [HIT], ["x:p9:c0"])
    reasons = qa_review_reasons(result)
    assert len(reasons) == 3 and "confidence: low" in reasons[0]


def test_escalation_is_listed_first():
    result = AgentResult("q", "Escalated", escalated=True, ticket="ESC-1", review_reasons=["no cited evidence"])
    assert agent_review_reasons(result)[0].startswith("escalated") and len(agent_review_reasons(result)) == 2


def test_app_renders_without_calling_models():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Ask a policy question", "Review a claim form", "Ask the agent"]
    assert len(at.selectbox[0].options) == 6  # the synthetic claim forms
