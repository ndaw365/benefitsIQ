"""Tests for the structured-output schemas (no API calls)."""

import pytest
from pydantic import ValidationError

from src.schemas import PolicyAnswer

VALID = {
    "answer": "The elimination period is 90 days.",
    "confidence": "high",
    "sources": ["p.12"],
    "needs_human_review": False,
}


def test_valid_answer_parses() -> None:
    result = PolicyAnswer.model_validate(VALID)
    assert result.confidence == "high"
    assert result.sources == ["p.12"]


def test_confidence_outside_allowed_values_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyAnswer.model_validate({**VALID, "confidence": "very high"})


def test_sources_is_required() -> None:
    missing = {k: v for k, v in VALID.items() if k != "sources"}
    with pytest.raises(ValidationError):
        PolicyAnswer.model_validate(missing)


def test_empty_sources_is_allowed() -> None:
    assert PolicyAnswer.model_validate({**VALID, "sources": []}).sources == []


def test_json_schema_marks_every_field_required() -> None:
    required = set(PolicyAnswer.model_json_schema()["required"])
    assert required == {"answer", "confidence", "sources", "needs_human_review"}
