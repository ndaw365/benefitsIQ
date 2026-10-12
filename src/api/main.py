"""BenefitsIQ HTTP API: the same RAG, extraction, and agent code the UI uses, as JSON endpoints.

Run from the repo root:  python -m uvicorn src.api.main:app --reload
Interactive docs:        http://127.0.0.1:8000/docs  (click "Authorize" and paste BENEFITSIQ_API_KEY)
"""

import os
from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.genai import errors

from src.agent.loop import log_run, run_agent
from src.agent.tools import AgentTools
from src.extraction.extract import process_claim
from src.llm import LLMError, ToolChat, ask
from src.api.security import RateLimiter, require_api_key
from src.rag.answer import answer_question
from src.rag.retrieve import IndexNotBuiltError
from src.schemas import (
    AgentRequest, AgentResponse, AgentStep, AskRequest, AskResponse, ExtractRequest, ExtractResponse,
    ReviewIssue, SourceRef,
)
from src.views import agent_review_reasons, qa_review_reasons, source_label

app = FastAPI(
    title="BenefitsIQ",
    description="Claims and policy copilot (portfolio prototype). Public sample policies and synthetic claims "
    "only. Outputs are estimates for an examiner to verify, never claim decisions.",
    version="0.1.0",
)


# Rate-limit budget per API key per minute. An agent request can make up to 7 model calls, so it costs more.
limiter = RateLimiter(limit=int(os.getenv("BENEFITSIQ_RATE_LIMIT", "20")))
COST = {"ask": 1, "extract": 1, "agent": 5}


def guard(endpoint: str) -> Callable:
    """Dependency: valid API key, then charge this endpoint's cost against the key's budget."""
    def check(key: str = Depends(require_api_key)) -> None:
        limiter.check(key, COST[endpoint])
    return check


@app.exception_handler(IndexNotBuiltError)
def index_not_built(request: Request, exc: IndexNotBuiltError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --- Dependencies: real implementations here, swapped for fakes in tests -----------------

@lru_cache(maxsize=1)
def get_resources():
    """Embedding model and Chroma collection, loaded once per process on first request."""
    from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, make_embedder
    from src.rag.retrieve import open_collection
    return make_embedder(), open_collection(DEFAULT_SIZE, DEFAULT_OVERLAP)


def get_llm() -> Callable:
    return ask


def get_chat_factory() -> Callable:
    return ToolChat


def model_failure(exc: Exception) -> HTTPException:
    """Map model and upstream failures to honest HTTP statuses instead of a generic 500."""
    if isinstance(exc, LLMError):  # truncated, empty, or schema-violating output
        return HTTPException(502, f"Model returned unusable output: {exc}")
    if isinstance(exc, errors.APIError) and exc.code == 429:  # still rate-limited after ask()'s retries
        return HTTPException(429, "Model rate limit reached; retry later.")
    return HTTPException(503, f"Model service unavailable: {type(exc).__name__}")


# --- Endpoints ----------------------------------------------------------------------------
# Plain `def`, not `async def`: our model and SDK calls block, and FastAPI runs sync
# endpoints in a thread pool. Blocking inside `async def` would stall every other request.

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard("ask"))])
def ask_policy(req: AskRequest, resources=Depends(get_resources), llm=Depends(get_llm)) -> AskResponse:
    embed, collection = resources
    try:
        result = answer_question(req.question, embed, collection, llm=llm)
    except (LLMError, errors.APIError) as exc:
        raise model_failure(exc) from exc
    by_id = {h.chunk_id: h for h in result.hits}
    sources = [SourceRef(chunk_id=s, doc_id=by_id[s].doc_id, page=by_id[s].page,
                         label=source_label(by_id[s].doc_id, by_id[s].page))
               for s in result.answer.sources if s in by_id]
    reasons = qa_review_reasons(result)
    return AskResponse(answer=result.answer, sources=sources, needs_review=bool(reasons), review_reasons=reasons)


@app.post("/claims/extract", response_model=ExtractResponse, dependencies=[Depends(guard("extract"))])
def extract_claim(req: ExtractRequest, llm=Depends(get_llm)) -> ExtractResponse:
    try:
        result = process_claim(req.form_text, llm)
    except (LLMError, errors.APIError) as exc:
        raise model_failure(exc) from exc
    return ExtractResponse(claim=result.claim, issues=[ReviewIssue(code=i.code, detail=i.detail) for i in result.issues],
                           needs_review=result.needs_review)


@app.post("/agent", response_model=AgentResponse, dependencies=[Depends(guard("agent"))])
def ask_agent(req: AgentRequest, resources=Depends(get_resources), llm=Depends(get_llm),
              chat_factory=Depends(get_chat_factory)) -> AgentResponse:
    embed, collection = resources
    tools = AgentTools(embed, collection, extract=lambda text: process_claim(text, llm))
    try:
        result = run_agent(req.question, tools, chat_factory=chat_factory)
    except (LLMError, errors.APIError) as exc:
        raise model_failure(exc) from exc
    log_run(result)
    reasons = agent_review_reasons(result)
    return AgentResponse(
        answer=result.answer, steps=[AgentStep(tool=s.tool, args=s.args, output=s.output) for s in result.steps],
        escalated=result.escalated, ticket=result.ticket, cited_ids=result.cited_ids,
        needs_review=result.needs_review, review_reasons=reasons,
    )
