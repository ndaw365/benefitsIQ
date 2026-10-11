"""End-to-end eval: answer every golden question, then grade it with an LLM judge."""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.eval.golden import load_golden
from src.eval.retrieval import score_item
from src.llm import LLMError, ask
from src.rag.answer import RagResult, answer_question, build_prompt
from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, make_embedder
from src.rag.load import load_all
from src.rag.retrieve import open_collection
from src.schemas import GoldenItem, JudgeVerdict

RESULTS_DIR = Path("results/faithfulness")

JUDGE_SYSTEM = """You are a strict evaluator of answers about insurance policy documents.
You receive the passages an assistant was given, the question, a reference answer written by a human
expert, and the assistant's answer. Judge two things independently:

1. faithfulness: compare the assistant's answer to the PASSAGES only. Ignore the reference answer here.
   A claim is supported only if the passages state it; plausible general knowledge does not count.
2. correctness: compare the assistant's answer to the REFERENCE answer. Key facts are numbers, durations,
   percentages, conditions, and which policy they belong to. Extra correct detail is fine.

Be literal. Do not give credit for an answer that is vague where the reference is specific."""


@dataclass(frozen=True)
class FaithResult:
    id: str
    category: str
    span_hit: bool  # did retrieval surface the labeled quote? Always False for unanswerable items
    answer: str
    confidence: str
    needs_review: bool
    sources: list[str]
    invalid_sources: list[str]
    faithfulness: str
    correctness: str
    unsupported_claims: list[str]
    reason: str
    outcome: str


def build_judge_prompt(item: GoldenItem, result: RagResult) -> str:
    """Show the judge exactly what the assistant saw, plus the reference and the assistant's answer."""
    seen = build_prompt(item.question, result.hits) if result.hits else f"Passages: (none)\n\nQuestion: {item.question}"
    return f"{seen}\n\nReference answer: {item.expected_answer}\n\nAssistant answer: {result.answer.answer}"


def classify(category: str, correctness: str, needs_review: bool) -> str:
    """Turn a verdict into the outcome the business cares about.

    unsafe_error is the critical one: a wrong or incomplete answer that would reach a user unflagged.
    """
    if category == "unanswerable":
        if correctness == "abstained":
            return "correct"
        return "flagged_error" if needs_review else "unsafe_error"
    if correctness == "correct":
        return "correct"
    if correctness == "abstained":
        return "safe_abstain"  # missed the answer, but said so
    return "flagged_error" if needs_review else "unsafe_error"


def evaluate_item(
    item: GoldenItem, rag: Callable[[str], RagResult], pages: dict, llm: Callable[..., JudgeVerdict] = ask
) -> FaithResult:
    """Answer one golden question, judge it, and classify the outcome."""
    result = rag(item.question)
    verdict = llm(build_judge_prompt(item, result), system=JUDGE_SYSTEM, schema=JudgeVerdict)
    span_hit = score_item(item, result.hits, pages).span_hit if item.sources else False
    # Judges don't always follow label instructions; enforce the rule that an abstention makes no claims.
    faithfulness = "no_claims" if verdict.correctness == "abstained" else verdict.faithfulness
    a = result.answer
    return FaithResult(
        id=item.id, category=item.category, span_hit=span_hit, answer=a.answer, confidence=a.confidence,
        needs_review=a.needs_human_review, sources=a.sources, invalid_sources=result.invalid_sources,
        faithfulness=faithfulness, correctness=verdict.correctness,
        unsupported_claims=verdict.unsupported_claims, reason=verdict.reason,
        outcome=classify(item.category, verdict.correctness, a.needs_human_review),
    )


def summarize(results: list[FaithResult]) -> dict:
    """Outcome counts, faithfulness over answers that make claims, and retrieval-vs-generation split."""
    with_claims = [r for r in results if r.faithfulness != "no_claims"]
    answerable = [r for r in results if r.category != "unanswerable"]
    return {
        "n": len(results),
        "outcomes": dict(Counter(r.outcome for r in results)),
        "faithfulness_rate": round(sum(r.faithfulness == "supported" for r in with_claims) / len(with_claims), 3)
        if with_claims else None,
        "faithfulness": dict(Counter(r.faithfulness for r in results)),
        "correctness": dict(Counter(r.correctness for r in results)),
        # Answerable items only: where did correct answers come from, and where did failures start?
        "retrieval_x_correct": dict(Counter(
            f"{'hit' if r.span_hit else 'miss'}/{r.correctness}" for r in answerable
        )),
        "invented_citations": sum(bool(r.invalid_sources) for r in results),
    }


def load_done(path: Path) -> dict[str, dict]:
    """Rows already written by an earlier (possibly interrupted) run, keyed by id."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness + correctness eval with an LLM judge.")
    parser.add_argument("--limit", type=int, help="only the first N golden items (smoke test)")
    parser.add_argument("--delay", type=float, default=8.0, help="seconds between items (2 LLM calls each)")
    parser.add_argument("--out", default="baseline", help="run name; results go to results/faithfulness/<out>.jsonl")
    args = parser.parse_args()

    items = load_golden()[: args.limit]
    pages = {(p.doc_id, p.page): p.text for p in load_all()}
    embed, collection = make_embedder(), open_collection(DEFAULT_SIZE, DEFAULT_OVERLAP)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.out}.jsonl"
    done = load_done(path)  # resume: free-tier runs can die halfway on rate limits

    for n, item in enumerate(items, start=1):
        if item.id in done:
            continue
        try:
            row = evaluate_item(item, lambda q: answer_question(q, embed, collection), pages)
        except LLMError as exc:  # truncated/empty/malformed output: record it, don't crash the run
            print(f"[{n}/{len(items)}] {item.id}: ERROR {exc}")
            continue
        with path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        done[item.id] = asdict(row)
        print(f"[{n}/{len(items)}] {row.outcome:13} faith={row.faithfulness:19} {item.id}")
        time.sleep(args.delay)

    results = [FaithResult(**done[i.id]) for i in items if i.id in done]
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
