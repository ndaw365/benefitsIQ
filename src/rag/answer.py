"""Answer a policy question from retrieved chunks, with verified citations."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from src.llm import ask
from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, EmbedFn, make_embedder
from src.rag.retrieve import Hit, open_collection, retrieve
from src.rag.route import detect_policies
from src.schemas import PolicyAnswer

SYSTEM = """You help insurance claims examiners answer questions about group benefit policies.
Rules:
- Answer ONLY from the passages provided. Do not use outside knowledge about insurance.
- Put the ID of every passage that supports your answer in `sources`, copied exactly as shown in [brackets].
- If the passages do not contain the answer, say so plainly, use confidence "low", leave sources empty,
  and set needs_human_review to true.
- If passages from different policies give different answers, state each answer with its policy
  and set needs_human_review to true."""

NOT_FOUND = PolicyAnswer(
    answer="No relevant policy passages were found for this question.",
    confidence="low",
    sources=[],
    needs_human_review=True,
)


@dataclass(frozen=True)
class RagResult:
    answer: PolicyAnswer
    hits: list[Hit]  # what the model saw, so evals can separate retrieval from generation errors
    invalid_sources: list[str]  # cited IDs that were not among the hits (removed from answer.sources)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Format the hits as labeled passages followed by the question."""
    passages = "\n\n".join(f"[{h.chunk_id}] ({h.doc_id}, page {h.page})\n{h.text}" for h in hits)
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def check_citations(answer: PolicyAnswer, hits: list[Hit]) -> tuple[PolicyAnswer, list[str]]:
    """Drop cited IDs the model was never shown; flag the answer for review if any were dropped."""
    shown = {h.chunk_id for h in hits}
    valid = [s for s in answer.sources if s in shown]
    invalid = [s for s in answer.sources if s not in shown]
    if not invalid:
        return answer, []
    return answer.model_copy(update={"sources": valid, "needs_human_review": True}), invalid


def answer_question(
    question: str, embed: EmbedFn, collection: Collection, k: int = 5,
    llm: Callable[..., PolicyAnswer] = ask,
) -> RagResult:
    """Route to named policies, retrieve, generate a PolicyAnswer, and verify its citations."""
    hits = retrieve(question, embed, collection, k, detect_policies(question))
    if not hits:
        return RagResult(NOT_FOUND, [], [])  # no context: don't let the model answer from memory
    raw = llm(build_prompt(question, hits), system=SYSTEM, schema=PolicyAnswer)
    answer, invalid = check_citations(raw, hits)
    return RagResult(answer, hits, invalid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a policy question with citations.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    result = answer_question(
        args.question, make_embedder(), open_collection(args.size, args.overlap), args.k
    )
    a = result.answer
    print(f"\nAnswer: {a.answer}")
    print(f"Confidence: {a.confidence}   Needs human review: {a.needs_human_review}")
    print(f"Sources: {a.sources or 'none'}")
    if result.invalid_sources:
        print(f"Removed invented citations: {result.invalid_sources}")
    print("\nRetrieved:", ", ".join(f"{h.chunk_id} ({h.distance:.3f})" for h in result.hits) or "none")


if __name__ == "__main__":
    main()
