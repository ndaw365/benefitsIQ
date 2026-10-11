"""Claim review rules, checked against the hand labels and on targeted edge cases."""

import json
from pathlib import Path

import pytest

from src.extraction.review import review_claim
from src.schemas import ClaimExtraction

LABELS = [json.loads(line) for line in Path("data/claims/labels.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("label", LABELS, ids=[row["id"] for row in LABELS])
def test_labels_are_valid_and_rules_find_planted_issues(label):
    claim = ClaimExtraction.model_validate(label["expected"])  # labels must fit the schema
    assert sorted(i.code for i in review_claim(claim)) == sorted(label["expected_issues"])


def test_every_label_has_a_form():
    assert all(Path(f"data/claims/{row['id']}.txt").exists() for row in LABELS)


def clean_claim(**changes) -> ClaimExtraction:
    return ClaimExtraction.model_validate({**LABELS[0]["expected"], **changes})


def test_bad_date_format_is_an_issue_not_a_crash():
    assert [i.code for i in review_claim(clean_claim(date_of_birth="04/12/1971"))] == ["bad_date:date_of_birth"]


def test_implausible_age():
    assert [i.code for i in review_claim(clean_claim(date_of_birth="2019-01-01"))] == ["implausible_age"]


def test_unclear_signature_needs_review():
    assert [i.code for i in review_claim(clean_claim(employee_signed=None))] == ["unsigned"]


def test_small_salary_gap_is_tolerated():
    assert review_claim(clean_claim(annual_salary=66000)) == []  # 1.9% above 12 x 5,400
