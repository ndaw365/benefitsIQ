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


class JudgeVerdict(BaseModel):
    """LLM-as-judge grading of one RAG answer. Faithfulness and correctness are judged separately."""

    reason: str = Field(description="Two or three sentences explaining both labels.")
    unsupported_claims: list[str] = Field(
        description="Factual claims in the assistant's answer that the passages do not state. Empty if none."
    )
    faithfulness: Literal["supported", "partially_supported", "unsupported", "no_claims"] = Field(
        description="Assistant answer vs. the PASSAGES only (ignore the reference answer). "
        "no_claims: the answer only says the information is not available."
    )
    correctness: Literal["correct", "partially_correct", "incorrect", "abstained"] = Field(
        description="Assistant answer vs. the REFERENCE answer. abstained: the answer says the information "
        "is not available, whatever the reference says."
    )


NULL_RULE = " Null if the form does not state it or it is illegible; never infer or guess."


class OtherIncome(BaseModel):
    """One other-income source listed on a claim form (offsets the disability benefit)."""

    source: str = Field(description="Type of income as written, e.g. 'Social Security Disability'.")
    monthly_amount: float | None = Field(description="Dollars per month as a number." + NULL_RULE)


class ClaimExtraction(BaseModel):
    """Fields extracted from a long-term disability claim form (employee statement)."""

    claimant_name: str | None = Field(description="Employee's full name." + NULL_RULE)
    employer: str | None = Field(description="Policyholder or employer name as written." + NULL_RULE)
    policy_number: str | None = Field(description="Policy or group policy number as written." + NULL_RULE)
    date_of_birth: str | None = Field(description="YYYY-MM-DD." + NULL_RULE)
    last_day_worked: str | None = Field(description="YYYY-MM-DD." + NULL_RULE)
    disability_onset_date: str | None = Field(description="Date disability began, YYYY-MM-DD." + NULL_RULE)
    date_signed: str | None = Field(description="Date the employee signed, YYYY-MM-DD." + NULL_RULE)
    diagnosis: str | None = Field(description="Condition as described on the form." + NULL_RULE)
    hours_per_week: float | None = Field(description="Hours worked per week as a number." + NULL_RULE)
    monthly_earnings: float | None = Field(description="Basic monthly earnings in dollars, as a number." + NULL_RULE)
    annual_salary: float | None = Field(description="Annual salary in dollars, as a number." + NULL_RULE)
    other_income: list[OtherIncome] = Field(description="Other income sources listed. Empty list if none.")
    employee_signed: bool | None = Field(description="True if a signature is present, false if the line is blank.")
    physician_statement_attached: bool | None = Field(description="Whether the attending physician statement is attached." + NULL_RULE)
