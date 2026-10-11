"""Business-rule review of an extracted claim: collect every reason a human should look at it."""

from dataclasses import dataclass
from datetime import date

from src.schemas import ClaimExtraction

REQUIRED = (
    "claimant_name", "policy_number", "date_of_birth", "last_day_worked",
    "disability_onset_date", "diagnosis", "monthly_earnings",
)
DATE_FIELDS = ("date_of_birth", "last_day_worked", "disability_onset_date", "date_signed")
MIN_HOURS_PER_WEEK = 30  # Lincoln p.3 "MINIMUM HOURS: 30 hours per week"; Standard p.6 "at least 30 hours each week"
SALARY_TOLERANCE = 0.05  # annual vs 12 x monthly; small gaps are rounding or a mid-year raise
AGE_RANGE = (16, 80)  # outside this, a date was almost certainly misread


@dataclass(frozen=True)
class Issue:
    code: str  # stable, comparable in tests and evals, e.g. "missing:policy_number"
    detail: str  # human-readable explanation for the examiner


def parse_dates(claim: ClaimExtraction) -> tuple[dict[str, date], list[Issue]]:
    """Parse the YYYY-MM-DD fields we were promised; a bad format is an issue, not a crash."""
    dates, issues = {}, []
    for field in DATE_FIELDS:
        value = getattr(claim, field)
        if value is None:
            continue
        try:
            dates[field] = date.fromisoformat(value)
        except ValueError:
            issues.append(Issue(f"bad_date:{field}", f"{field}={value!r} is not YYYY-MM-DD"))
    return dates, issues


def review_claim(claim: ClaimExtraction) -> list[Issue]:
    """Return every rule the claim breaks; an empty list means no human review is needed."""
    issues = [Issue(f"missing:{f}", f"{f} not found on the form") for f in REQUIRED if getattr(claim, f) is None]
    dates, date_issues = parse_dates(claim)
    issues += date_issues

    signed = dates.get("date_signed")
    for field in ("last_day_worked", "disability_onset_date"):
        if signed and field in dates and dates[field] > signed:
            issues.append(Issue("date_order", f"{field} {dates[field]} is after date_signed {signed}"))

    dob, onset = dates.get("date_of_birth"), dates.get("disability_onset_date")
    if dob and onset:
        age = onset.year - dob.year - ((onset.month, onset.day) < (dob.month, dob.day))
        if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
            issues.append(Issue("implausible_age", f"age {age} at disability onset"))

    if claim.hours_per_week is not None and claim.hours_per_week < MIN_HOURS_PER_WEEK:
        issues.append(Issue("below_min_hours", f"{claim.hours_per_week} hours/week < {MIN_HOURS_PER_WEEK}"))

    if claim.monthly_earnings and claim.annual_salary:
        gap = abs(claim.annual_salary - 12 * claim.monthly_earnings) / claim.annual_salary
        if gap > SALARY_TOLERANCE:
            issues.append(Issue("salary_mismatch",
                                f"annual ${claim.annual_salary:,.0f} vs 12 x monthly ${12 * claim.monthly_earnings:,.0f}"))

    if claim.employee_signed is not True:  # None (unclear) is treated like unsigned: a human checks
        issues.append(Issue("unsigned", "employee signature missing or unclear"))
    if claim.physician_statement_attached is not True:
        issues.append(Issue("no_physician_statement", "attending physician statement not confirmed"))
    return issues
