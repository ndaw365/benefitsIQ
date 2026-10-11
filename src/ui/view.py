"""Pure formatting helpers for the UI, kept out of the Streamlit script so they can be unit tested."""

from src.agent.loop import AgentResult
from src.rag.answer import RagResult
from src.rag.load import DOC_TITLES


def source_label(doc_id: str, page: int) -> str:
    """'Lincoln Financial Group ... Moravian University, page 3' instead of an internal id."""
    return f"{DOC_TITLES.get(doc_id, doc_id)}, page {page}"


def qa_review_reasons(result: RagResult) -> list[str]:
    """Why a policy answer needs a human, in words an examiner can act on."""
    reasons = []
    if not result.hits:
        reasons.append("no policy passages matched the question")
    if result.answer.needs_human_review:
        reasons.append(f"the model flagged its own answer (confidence: {result.answer.confidence})")
    if result.invalid_sources:
        reasons.append(f"removed citations to passages it was never shown: {result.invalid_sources}")
    if result.hits and not result.answer.sources:
        reasons.append("the answer cites no passage")
    return reasons


def cited_sources(result: RagResult) -> list[str]:
    """Readable labels for the passages the answer cites, in citation order, without duplicates."""
    by_id = {h.chunk_id: h for h in result.hits}
    labels = [source_label(by_id[s].doc_id, by_id[s].page) for s in result.answer.sources if s in by_id]
    return list(dict.fromkeys(labels))


def agent_review_reasons(result: AgentResult) -> list[str]:
    reasons = list(result.review_reasons)
    if result.escalated:
        reasons.insert(0, f"escalated to a human examiner (ticket {result.ticket})")
    return reasons
