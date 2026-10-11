"""Field-level extraction accuracy and issue detection against hand-labeled claim forms."""

import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from src.extraction.extract import process_claim
from src.llm import LLMError, ask
from src.schemas import ClaimExtraction

CLAIMS_DIR = Path("data/claims")
RESULTS_DIR = Path("results/extraction")


def normalize_text(value: str) -> str:
    """Case, punctuation, and spacing differences aren't extraction errors."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def field_matches(got, expected) -> bool:
    """Compare one extracted field with its label. None must match None exactly."""
    if got is None or expected is None:
        return got is expected
    if isinstance(expected, bool) or isinstance(got, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        return abs(float(got) - float(expected)) < 0.01
    if isinstance(expected, list):  # other_income: same count and same amounts; source wording not graded
        amounts = lambda rows: sorted((r["monthly_amount"] is None, r["monthly_amount"] or 0) for r in rows)
        return len(got) == len(expected) and amounts(got) == amounts(expected)
    return normalize_text(str(got)) == normalize_text(str(expected))


def score_claim(claim: ClaimExtraction, issues: list[str], label: dict) -> dict:
    """Per-field matches and issue precision/recall for one form."""
    got = claim.model_dump()
    wrong = {f: {"got": got[f], "expected": e} for f, e in label["expected"].items() if not field_matches(got[f], e)}
    expected_issues, found = set(label["expected_issues"]), set(issues)
    return {
        "id": label["id"],
        "fields_total": len(label["expected"]),
        "fields_wrong": wrong,
        "issues_expected": sorted(expected_issues),
        "issues_found": sorted(found),
        "issues_missed": sorted(expected_issues - found),
        "issues_spurious": sorted(found - expected_issues),
        "review_correct": bool(found) == bool(expected_issues),
    }


def summarize(rows: list[dict]) -> dict:
    total = sum(r["fields_total"] for r in rows)
    wrong = sum(len(r["fields_wrong"]) for r in rows)
    expected = sum(len(r["issues_expected"]) for r in rows)
    found = sum(len(r["issues_found"]) for r in rows)
    missed = sum(len(r["issues_missed"]) for r in rows)
    spurious = sum(len(r["issues_spurious"]) for r in rows)
    return {
        "forms": len(rows),
        "field_accuracy": round(1 - wrong / total, 3) if total else None,
        "issue_recall": round(1 - missed / expected, 3) if expected else None,  # planted issues we caught
        "issue_precision": round(1 - spurious / found, 3) if found else None,  # flags that were real
        "review_decision_accuracy": round(sum(r["review_correct"] for r in rows) / len(rows), 3) if rows else None,
    }


def evaluate(labels: list[dict], llm: Callable[..., ClaimExtraction] = ask, delay: float = 0.0) -> list[dict]:
    rows = []
    for label in labels:
        try:
            result = process_claim((CLAIMS_DIR / f"{label['id']}.txt").read_text(), llm)
        except LLMError as exc:  # a failed extraction is reported, not silently skipped
            print(f"{label['id']}: ERROR {exc}")
            continue
        rows.append(score_claim(result.claim, [i.code for i in result.issues], label))
        time.sleep(delay)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction eval on the labeled synthetic claims.")
    parser.add_argument("--out", default="baseline")
    parser.add_argument("--delay", type=float, default=4.0)
    args = parser.parse_args()

    labels = [json.loads(line) for line in (CLAIMS_DIR / "labels.jsonl").read_text().splitlines() if line.strip()]
    rows = evaluate(labels, delay=args.delay)
    for r in rows:
        status = "ok " if not r["fields_wrong"] and r["review_correct"] and not r["issues_missed"] else "ERR"
        print(f"\n{status} {r['id']}  issues found={r['issues_found']}  missed={r['issues_missed']}  spurious={r['issues_spurious']}")
        for field, diff in r["fields_wrong"].items():
            print(f"     {field}: got {diff['got']!r}  expected {diff['expected']!r}")
    summary = summarize(rows)
    print("\n" + json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.out}.json").write_text(json.dumps({"summary": summary, "forms": rows}, indent=2))


if __name__ == "__main__":
    main()
