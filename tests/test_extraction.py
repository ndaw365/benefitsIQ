"""Extraction plumbing and scoring with a fake LLM that returns the hand labels."""

import json
from pathlib import Path

from src.eval.extraction import evaluate, field_matches, summarize
from src.extraction.extract import process_claim
from src.schemas import ClaimExtraction

LABELS = [json.loads(line) for line in Path("data/claims/labels.jsonl").read_text().splitlines()]


def test_field_matching_rules():
    assert field_matches(None, None) and not field_matches(None, "x") and not field_matches("x", None)
    assert field_matches(5400.0, 5400) and not field_matches(5400.5, 5400)
    assert field_matches("Lumbar disc herniation.", "lumbar  disc herniation")
    assert not field_matches(False, None) and field_matches(True, True)
    assert field_matches([{"source": "SSDI", "monthly_amount": 1850}],
                         [{"source": "Social Security Disability", "monthly_amount": 1850.0}])
    assert not field_matches([], [{"source": "SSDI", "monthly_amount": None}])


def test_process_claim_reviews_what_the_llm_returned():
    label = LABELS[2]  # claim_03: onset date after signature
    result = process_claim("form text", llm=lambda prompt, **kw: ClaimExtraction.model_validate(label["expected"]))
    assert result.needs_review and [i.code for i in result.issues] == ["date_order"]


def test_perfect_extractor_scores_perfectly():
    labels_by_text = {Path(f"data/claims/{row['id']}.txt").read_text(): row for row in LABELS}

    def oracle(prompt, **kwargs):
        form_text = prompt.split("Claim form:\n\n", 1)[1]
        return ClaimExtraction.model_validate(labels_by_text[form_text]["expected"])

    summary = summarize(evaluate(LABELS, llm=oracle))
    assert summary == {"forms": 6, "field_accuracy": 1.0, "issue_recall": 1.0,
                       "issue_precision": 1.0, "review_decision_accuracy": 1.0}
