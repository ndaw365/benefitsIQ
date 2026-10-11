"""The agent loop: model requests tools, our code runs them, guardrails check the final answer."""

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.agent.tools import AgentTools
from src.llm import ToolChat, ToolSpec, ToolTurn

RUN_LOG = Path("logs/agent_runs.jsonl")
MAX_STEPS = 6  # model turns per question; past this, a human takes over
CITATION = re.compile(r"\[([a-z0-9_]+:p\d+:c\d+)\]")
DECISION_WORDS = re.compile(r"\b(approved?|approving|den(y|ied|ial))\b", re.IGNORECASE)

SYSTEM = """You assist insurance claims examiners with group benefit policies and claim forms.
Tools: get_claim, search_policy, calculate_benefit, escalate_to_human.
Rules:
- Every policy fact (percentages, maximums, minimums, periods, conditions) must come from search_policy
  results in this conversation. Never use outside knowledge about insurance.
- Do all benefit arithmetic with calculate_benefit, using inputs from the claim and the policy passages.
- In your final answer, cite the passage id in [brackets] after each policy fact, e.g. [lincoln_moravian_ltd:p3:c0].
- You never approve or deny claims. If asked for a claim decision, or if the documents or claim data are
  missing or contradictory, call escalate_to_human.
- If the documents do not answer the question, say so plainly; do not guess.
- Final answers: 2-6 sentences for a claims examiner. Say benefit figures are estimates."""


@dataclass
class Step:
    tool: str
    args: dict
    output: dict


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    escalated: bool = False
    ticket: str | None = None
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.escalated or bool(self.review_reasons)


def check_answer(answer: str, steps: list[Step], seen_ids: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Guardrails on a final answer. Returns (cited ids, invented citations, reasons a human should review)."""
    cited = CITATION.findall(answer)
    invalid = [c for c in cited if c not in seen_ids]
    reasons = []
    if invalid:
        reasons.append(f"cites passages it never retrieved: {invalid}")
    if not set(cited) - set(invalid):
        reasons.append("no cited evidence for the answer")
    tools_used = [s.tool for s in steps]
    if "calculate_benefit" in tools_used and (
        "search_policy" not in tools_used or tools_used.index("calculate_benefit") < tools_used.index("search_policy")
    ):
        reasons.append("benefit calculated before searching the policy for its inputs")
    if DECISION_WORDS.search(answer):
        reasons.append("answer uses claim-decision language")
    for step in steps:  # rule-based claim issues stay flagged, whatever the answer says about them
        if step.tool == "get_claim" and step.output.get("review_issues"):
            reasons.append(f"claim {step.args.get('claim_id')} has review issues: {step.output['review_issues']}")
    return cited, invalid, reasons


def run_agent(
    question: str,
    tools: AgentTools,
    chat_factory: Callable[[str, list[ToolSpec]], ToolChat] = ToolChat,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Loop until the model answers, escalates, or hits the step limit (which escalates for it)."""
    chat = chat_factory(SYSTEM, tools.specs)
    result = AgentResult(question=question, answer="")
    turn: ToolTurn = chat.send(question)

    for _ in range(max_steps):
        if not turn.calls:  # the model answered
            result.answer = turn.text or ""
            result.cited_ids, result.invalid_citations, result.review_reasons = check_answer(
                result.answer, result.steps, tools.seen_ids)
            return result
        outputs = []
        for call in turn.calls:
            output = tools.run(call)
            result.steps.append(Step(call.name, call.args, output))
            outputs.append((call, output))
        if tools.escalation:  # a human owns the case now; stop spending calls
            result.escalated, result.ticket = True, tools.escalation["ticket"]
            result.answer = f"Escalated to a human examiner ({result.ticket}): {tools.escalation['reason']}"
            return result
        turn = chat.send_tool_results(outputs)

    tools.escalate_to_human(reason=f"agent hit the {max_steps}-step limit", summary=question)
    result.escalated, result.ticket = True, tools.escalation["ticket"]
    result.answer = f"Escalated to a human examiner ({result.ticket}): step limit reached."
    return result


def log_run(result: AgentResult, path: Path = RUN_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **asdict(result), "needs_review": result.needs_review}
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, make_embedder
    from src.rag.retrieve import open_collection

    parser = argparse.ArgumentParser(description="Ask the claims agent a question.")
    parser.add_argument("question")
    args = parser.parse_args()

    tools = AgentTools(make_embedder(), open_collection(DEFAULT_SIZE, DEFAULT_OVERLAP))
    result = run_agent(args.question, tools)
    log_run(result)
    for n, step in enumerate(result.steps, start=1):
        brief = {k: v for k, v in step.output.items() if k != "passages"}
        if "passages" in step.output:
            brief["passages"] = [p["id"] for p in step.output["passages"]]
        print(f"step {n}: {step.tool}({json.dumps(step.args)})\n        -> {json.dumps(brief, default=str)[:300]}")
    print(f"\nANSWER: {result.answer}")
    print(f"needs_review={result.needs_review}  escalated={result.escalated}  cited={result.cited_ids}")
    for reason in result.review_reasons:
        print(f"  review: {reason}")


if __name__ == "__main__":
    main()
