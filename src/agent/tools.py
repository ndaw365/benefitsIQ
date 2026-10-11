"""The agent's tools: the specs the model sees, and the plain-Python code that runs them."""

import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from chromadb.api.models.Collection import Collection

from src.agent.benefit import calculate_benefit
from src.extraction.extract import process_claim
from src.llm import ToolCall, ToolSpec
from src.rag.index import EmbedFn
from src.rag.retrieve import retrieve
from src.rag.route import detect_policies

ESCALATION_LOG = Path("logs/escalations.jsonl")
CLAIMS_DIR = Path("data/claims")
SEARCH_K = 5  # passages per search
MAX_SKIPPED = 25  # bound on how far repeated searches dig, so a looping agent stays cheap
CLAIM_ID = re.compile(r"^claim_\d{2}$")  # strict allow-list: the id becomes a file path

GET_CLAIM = ToolSpec(
    name="get_claim",
    description="Read a claim form by id (e.g. 'claim_05'). Returns the extracted fields and any review "
    "issues found by business rules (missing fields, unsigned, date problems, below minimum hours).",
    parameters={
        "type": "object",
        "properties": {"claim_id": {"type": "string", "description": "Claim id like claim_05."}},
        "required": ["claim_id"],
    },
)

SEARCH_POLICY = ToolSpec(
    name="search_policy",
    description="Search the group benefit documents: Lincoln (Moravian University) LTD certificate, "
    "Standard Insurance (University of Alabama in Huntsville) LTD certificate, Minnesota Life (State of "
    "Colorado) term life certificate, and the DOL FMLA employee guide. Returns passages with ids to cite. "
    "Name the policy in the query when you know it.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A focused question about one topic."}},
        "required": ["query"],
    },
)

CALCULATE_BENEFIT = ToolSpec(
    name="calculate_benefit",
    description="Estimate a monthly LTD benefit with exact arithmetic: min(earnings x percent, maximum) minus "
    "other income, never below the policy minimum. Take the percent, maximum, and minimum from passages "
    "returned by search_policy; never assume them.",
    parameters={
        "type": "object",
        "properties": {
            "monthly_earnings": {"type": "number", "description": "Basic monthly earnings in dollars."},
            "benefit_percent": {"type": "number", "description": "Benefit percentage, e.g. 60 for 60%."},
            "max_monthly_benefit": {"type": "number", "description": "Policy maximum monthly benefit in dollars."},
            "other_income_monthly": {"type": "number", "description": "Monthly offsets such as Social Security Disability. 0 if none."},
            "min_monthly_benefit": {"type": "number", "description": "Flat minimum monthly benefit in dollars. 0 if none."},
            "min_percent_of_benefit": {"type": "number", "description": "Minimum as a percent of the benefit, e.g. 10. 0 if none."},
        },
        "required": ["monthly_earnings", "benefit_percent", "max_monthly_benefit"],
    },
)

ESCALATE_TO_HUMAN = ToolSpec(
    name="escalate_to_human",
    description="Hand the case to a human claims examiner. Use when the documents do not answer the question, "
    "inputs are missing or contradictory, or a wrong answer could affect a claim decision and you are unsure.",
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why a human is needed, in one sentence."},
            "summary": {"type": "string", "description": "What you found so far, with passage ids."},
        },
        "required": ["reason", "summary"],
    },
)


class AgentTools:
    """Runs tool calls for one agent session and remembers what it returned (for guardrails)."""

    specs = [GET_CLAIM, SEARCH_POLICY, CALCULATE_BENEFIT, ESCALATE_TO_HUMAN]

    def __init__(self, embed: EmbedFn, collection: Collection, escalation_log: Path = ESCALATION_LOG,
                 claims_dir: Path = CLAIMS_DIR, extract=process_claim):
        self._embed, self._collection, self._escalation_log = embed, collection, escalation_log
        self._claims_dir, self._extract = claims_dir, extract
        self.seen_ids: set[str] = set()  # passage ids returned this session; citations must come from here
        self.escalation: dict | None = None

    def run(self, call: ToolCall) -> dict:
        """Execute one call. Errors go back to the model as data so it can correct itself."""
        handlers = {
            "get_claim": self.get_claim,
            "search_policy": self.search_policy,
            "calculate_benefit": self.calculate_benefit,
            "escalate_to_human": self.escalate_to_human,
        }
        handler = handlers.get(call.name)
        if handler is None:
            return {"error": f"Unknown tool {call.name!r}. Available: {sorted(handlers)}"}
        try:
            return handler(**call.args)
        except (TypeError, ValueError) as exc:  # wrong/missing arguments, or impossible values
            return {"error": f"{call.name} failed: {exc}"}

    def get_claim(self, claim_id: str) -> dict:
        if not CLAIM_ID.match(claim_id):
            raise ValueError(f"invalid claim id {claim_id!r}; expected e.g. claim_05")
        path = self._claims_dir / f"{claim_id}.txt"
        if not path.exists():
            return {"error": f"No claim {claim_id}"}
        result = self._extract(path.read_text())
        return {"claim": result.claim.model_dump(), "review_issues": [i.detail for i in result.issues]}

    def search_policy(self, query: str) -> dict:
        """Top passages for the query, skipping any already returned this session.

        Retrieval is deterministic, so a reworded retry would otherwise return the same chunks;
        skipping them makes each retry dig further down the ranking. The model still has the
        earlier passages in its conversation history.
        """
        k = SEARCH_K + min(len(self.seen_ids), MAX_SKIPPED)
        hits = retrieve(query, self._embed, self._collection, k, detect_policies(query))
        fresh = [h for h in hits if h.chunk_id not in self.seen_ids][:SEARCH_K]
        skipped = sum(h.chunk_id in self.seen_ids for h in hits)
        self.seen_ids.update(h.chunk_id for h in fresh)
        out = {"passages": [{"id": h.chunk_id, "policy": h.doc_id, "page": h.page, "text": h.text} for h in fresh]}
        if skipped:
            out["note"] = f"Skipped {skipped} passage(s) already returned earlier in this conversation."
        if not fresh:
            out["note"] = "No new matching passages."
        return out

    def calculate_benefit(self, **kwargs) -> dict:
        return asdict(calculate_benefit(**kwargs))

    def escalate_to_human(self, reason: str, summary: str) -> dict:
        ticket = f"ESC-{uuid.uuid4().hex[:8]}"
        self.escalation = {"ticket": ticket, "reason": reason, "summary": summary,
                           "ts": datetime.now(timezone.utc).isoformat()}
        self._escalation_log.parent.mkdir(parents=True, exist_ok=True)
        with self._escalation_log.open("a") as f:
            f.write(json.dumps(self.escalation) + "\n")
        return {"status": "escalated", "ticket": ticket}
