"""Faithfulness eval plumbing with a fake RAG and a fake judge: no API calls."""

import pytest

from src.eval.faithfulness import (
    FaithResult, build_judge_prompt, classify, evaluate_item, finalize, load_done, summarize,
)
from src.rag.answer import RagResult
from src.rag.retrieve import Hit
from src.schemas import GoldenItem, JudgeVerdict, PolicyAnswer

PAGE = "Schedule. ELIMINATION PERIOD: 180 calendar days. More."
PAGES = {("doc", 3): PAGE}
ITEM = GoldenItem(id="elim", question="How long?", expected_answer="180 calendar days.", category="fact",
                  sources=[{"doc_id": "doc", "page": 3, "quote": "ELIMINATION PERIOD: 180 calendar days"}])
HIT = Hit("doc:p3:c0", "doc", 3, 0, PAGE, 0.2)
ANSWER = PolicyAnswer(answer="180 days.", confidence="high", sources=["doc:p3:c0"], needs_human_review=False)


@pytest.mark.parametrize("category, correctness, review, expected", [
    ("fact", "correct", False, "correct"),
    ("fact", "abstained", True, "safe_abstain"),
    ("fact", "incorrect", True, "flagged_error"),
    ("fact", "incorrect", False, "unsafe_error"),
    ("fact", "partially_correct", False, "unsafe_error"),
    ("unanswerable", "abstained", True, "correct"),
    ("unanswerable", "incorrect", False, "unsafe_error"),
    ("unanswerable", "correct", True, "flagged_error"),  # answered something it shouldn't, but flagged
])
def test_classify(category, correctness, review, expected):
    assert classify(category, correctness, review) == expected


def test_judge_prompt_shows_passages_reference_and_answer():
    prompt = build_judge_prompt(ITEM, RagResult(ANSWER, [HIT], []))
    assert "[doc:p3:c0]" in prompt and "Reference answer: 180 calendar days." in prompt
    assert prompt.endswith("Assistant answer: 180 days.")


def test_judge_prompt_with_no_passages():
    assert "Passages: (none)" in build_judge_prompt(ITEM, RagResult(ANSWER, [], []))


def test_evaluate_item_uses_judge_schema_and_scores_retrieval():
    calls = []

    def fake_judge(prompt, **kwargs):
        calls.append(kwargs)
        return JudgeVerdict(reason="ok", unsupported_claims=[], faithfulness="supported", correctness="correct")

    row = evaluate_item(ITEM, lambda q: RagResult(ANSWER, [HIT], []), PAGES, llm=fake_judge)
    assert calls[0]["schema"] is JudgeVerdict
    assert row.span_hit and row.outcome == "correct"


def make_row(**overrides) -> FaithResult:
    base = dict(id="x", category="fact", span_hit=True, answer="a", confidence="high", needs_review=False,
                sources=[], invalid_sources=[], faithfulness="supported", correctness="correct",
                unsupported_claims=[], reason="r", outcome="correct")
    return FaithResult(**{**base, **overrides})


def test_summary_faithfulness_excludes_abstentions():
    rows = [make_row(), make_row(id="y", faithfulness="unsupported", correctness="incorrect", outcome="unsafe_error"),
            make_row(id="z", faithfulness="no_claims", correctness="abstained", outcome="safe_abstain")]
    summary = summarize(rows)
    assert summary["faithfulness_rate"] == 0.5  # 1 of the 2 answers that made claims
    assert summary["outcomes"]["unsafe_error"] == 1
    assert summary["retrieval_x_correct"]["hit/abstained"] == 1


def test_load_done_resumes(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert set(load_done(path)) == {"a", "b"}
    assert load_done(tmp_path / "missing.jsonl") == {}


def test_abstention_is_normalized_to_no_claims():
    def judge_mislabels(prompt, **kwargs):  # what the real judge did in the smoke test
        return JudgeVerdict(reason="r", unsupported_claims=[], faithfulness="supported", correctness="abstained")

    row = evaluate_item(ITEM, lambda q: RagResult(ANSWER, [HIT], []), PAGES, llm=judge_mislabels)
    assert row.faithfulness == "no_claims"


def test_unanswerable_correct_means_abstained():
    row = finalize(make_row(category="unanswerable", correctness="correct", needs_review=True, outcome="flagged_error"))
    assert (row.correctness, row.faithfulness, row.outcome) == ("abstained", "no_claims", "correct")


def test_finalize_is_idempotent():
    row = make_row(correctness="partially_correct", needs_review=False)
    assert finalize(finalize(row)) == finalize(row)
