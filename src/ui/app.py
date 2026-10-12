"""BenefitsIQ: policy Q&A, claim review, and the claims agent.

Run from the repo root so `src` imports resolve:  python -m streamlit run src/ui/app.py
"""

from pathlib import Path

import streamlit as st

from src.llm import LLMError
from src.views import agent_review_reasons, cited_sources, qa_review_reasons, source_label

CLAIMS_DIR = Path("data/claims")

st.set_page_config(page_title="BenefitsIQ", layout="wide")
st.title("BenefitsIQ")
st.caption("Claims and policy copilot (portfolio prototype). Public sample policies and synthetic claims only. "
           "Answers and benefit figures are estimates for an examiner to verify, never claim decisions.")


@st.cache_resource(show_spinner="Loading embedding model and policy index...")
def resources():
    """Load once per server process, on first use, so opening the app is instant."""
    from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, make_embedder
    from src.rag.retrieve import open_collection
    return make_embedder(), open_collection(DEFAULT_SIZE, DEFAULT_OVERLAP)


def review_banner(reasons: list[str]) -> None:
    """Show why review is needed; a clean result still says to verify."""
    if reasons:
        st.error("**Needs human review**\n\n" + "\n".join(f"- {r}" for r in reasons))
    else:
        st.info("No review flags raised. Check the cited pages before relying on this answer.")


def guarded(action):
    """Run an LLM-backed action; show model or rate-limit failures instead of a stack trace."""
    try:
        return action()
    except LLMError as exc:
        st.error(f"The model returned unusable output: {exc}")
    except Exception as exc:  # e.g. quota exhausted after retries, network errors
        st.error(f"Request failed: {type(exc).__name__}: {str(exc)[:200]}")
    return None


ask_tab, claim_tab, agent_tab = st.tabs(["Ask a policy question", "Review a claim form", "Ask the agent"])

with ask_tab:
    question = st.text_input("Question", placeholder="What is the elimination period under the Moravian LTD policy?")
    if st.button("Answer", key="ask") and question.strip():
        from src.rag.answer import answer_question
        embed, collection = resources()
        with st.spinner("Retrieving passages and answering..."):
            st.session_state["qa"] = guarded(lambda: answer_question(question, embed, collection))
    result = st.session_state.get("qa")
    if result:
        review_banner(qa_review_reasons(result))
        st.markdown(f"**Answer:** {result.answer.answer}")
        st.caption(f"Confidence: {result.answer.confidence}")
        for label in cited_sources(result):
            st.markdown(f"- Source: {label}")
        with st.expander("Retrieved passages"):
            for hit in result.hits:
                st.markdown(f"**{source_label(hit.doc_id, hit.page)}** (`{hit.chunk_id}`, distance {hit.distance:.3f})")
                st.text(hit.text)

with claim_tab:
    claim_ids = sorted(p.stem for p in CLAIMS_DIR.glob("claim_*.txt"))
    claim_id = st.selectbox("Synthetic claim form", claim_ids)
    form_col, result_col = st.columns(2)
    with form_col:
        st.text((CLAIMS_DIR / f"{claim_id}.txt").read_text())
    with result_col:
        if st.button("Extract and review", key="extract"):
            from src.extraction.extract import process_claim
            with st.spinner("Extracting fields..."):
                st.session_state["claim"] = (claim_id, guarded(lambda: process_claim((CLAIMS_DIR / f"{claim_id}.txt").read_text())))
        saved = st.session_state.get("claim")
        if saved and saved[0] == claim_id and saved[1]:
            claim_result = saved[1]
            review_banner([i.detail for i in claim_result.issues])
            fields = claim_result.claim.model_dump()
            st.table({"field": list(fields), "value": [str(v) for v in fields.values()]})

with agent_tab:
    st.caption("Tools: get_claim, search_policy, calculate_benefit, escalate_to_human. Try: "
               "'What is the estimated monthly LTD benefit for claim_05?'")
    agent_question = st.text_input("Question for the agent", key="agent_q")
    if st.button("Run agent", key="agent") and agent_question.strip():
        from src.agent.loop import log_run, run_agent
        from src.agent.tools import AgentTools
        embed, collection = resources()
        with st.spinner("Agent working (several model calls)..."):
            agent_result = guarded(lambda: run_agent(agent_question, AgentTools(embed, collection)))
        if agent_result:
            log_run(agent_result)
        st.session_state["agent"] = agent_result
    agent_result = st.session_state.get("agent")
    if agent_result:
        review_banner(agent_review_reasons(agent_result))
        st.markdown(f"**Answer:** {agent_result.answer}")
        with st.expander(f"Tool calls ({len(agent_result.steps)})"):
            for n, step in enumerate(agent_result.steps, start=1):
                st.markdown(f"**{n}. {step.tool}**")
                st.json({"args": step.args, "output": step.output}, expanded=False)
