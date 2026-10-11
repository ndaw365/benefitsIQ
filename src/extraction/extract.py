"""Extract a claim form into a validated ClaimExtraction, then review it with business rules."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.extraction.review import Issue, review_claim
from src.llm import ask
from src.schemas import ClaimExtraction

SYSTEM = """You extract fields from long-term disability claim forms for an insurance claims team.
Rules:
- Copy values as the form states them. Do NOT correct apparent typos or inconsistencies
  (odd dates, salaries that don't add up); reviewers must see what the form actually says.
- Use null when a field is blank, missing, or illegible. Never infer a value from other fields.
- Write dates as YYYY-MM-DD. A two-digit year like '26' means 2026.
- Write money as a plain number of dollars (5400.0, not "$5,400.00")."""


@dataclass(frozen=True)
class ClaimResult:
    claim: ClaimExtraction
    issues: list[Issue]

    @property
    def needs_review(self) -> bool:
        return bool(self.issues)  # decided by rules, not by the model's confidence


def extract_claim(form_text: str, llm: Callable[..., ClaimExtraction] = ask) -> ClaimExtraction:
    """One schema-constrained LLM call: form text in, typed fields out (or an LLMError)."""
    return llm(f"Claim form:\n\n{form_text}", system=SYSTEM, schema=ClaimExtraction)


def process_claim(form_text: str, llm: Callable[..., ClaimExtraction] = ask) -> ClaimResult:
    """Extract, then review."""
    claim = extract_claim(form_text, llm)
    return ClaimResult(claim, review_claim(claim))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and review one claim form.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    result = process_claim(args.path.read_text())
    print(json.dumps(result.claim.model_dump(), indent=2))
    print(f"\nNeeds review: {result.needs_review}")
    for issue in result.issues:
        print(f"  - {issue.code}: {issue.detail}")


if __name__ == "__main__":
    main()
