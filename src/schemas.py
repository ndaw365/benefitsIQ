"""Pydantic schemas for every structured LLM output in BenefitsIQ."""

from typing import Literal

from pydantic import BaseModel, Field


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
