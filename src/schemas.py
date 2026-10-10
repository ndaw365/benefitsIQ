"""Pydantic schemas for every structured LLM output in BenefitsIQ."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PolicyAnswer(BaseModel):
    """What the assistant returns when answering a question about a policy."""

    answer: str = Field(
        description="Direct answer in at most 3 sentences, written for a claims examiner. "
        "If the information is not available, say so plainly instead of guessing."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How well the available information supports the answer."
    )
    sources: list[str] = Field(
        description="Identifiers of the passages that support the answer. "
        "Empty list if nothing supports it."
    )
    needs_human_review: bool = Field(
        description="True if the question is ambiguous, the information is missing, "
        "or a wrong answer could affect a claim decision."
    )


class GoldenSource(BaseModel):
    """Where a golden answer comes from: a page, plus a quote that proves it."""

    doc_id: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=10, description="Verbatim text from the cleaned page text.")


class GoldenItem(BaseModel):
    """One hand-checked question in the golden evaluation set."""

    id: str
    question: str
    expected_answer: str
    category: Literal["fact", "definition", "table", "multi_policy", "paraphrase", "unanswerable"]
    sources: list[GoldenSource]

    @model_validator(mode="after")
    def _sources_match_category(self) -> "GoldenItem":
        if (self.category == "unanswerable") != (not self.sources):
            raise ValueError("unanswerable items need no sources; every other item needs at least one")
        return self
