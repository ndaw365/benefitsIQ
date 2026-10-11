"""Deterministic LTD monthly benefit estimate. Plain Python: arithmetic is never left to the LLM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenefitEstimate:
    gross: float  # benefit percent x monthly earnings
    capped: float  # after the policy's maximum monthly benefit
    other_income: float  # offsets (e.g. Social Security Disability)
    minimum: float  # policy floor
    net: float  # estimated monthly benefit
    steps: list[str]  # the arithmetic, in words, so an examiner can check it


def calculate_benefit(
    monthly_earnings: float,
    benefit_percent: float,
    max_monthly_benefit: float,
    other_income_monthly: float = 0.0,
    min_monthly_benefit: float = 0.0,
    min_percent_of_benefit: float = 0.0,
) -> BenefitEstimate:
    """Estimate: min(earnings x percent, max) - other income, but never below the policy minimum.

    The minimum is the greater of a flat amount and a percent of the capped benefit, which is how
    the Lincoln schedule words it ("$100 or 10% of the ... Monthly Benefit, whichever is greater").
    An estimate only: actual claims follow each policy's full offset and earnings definitions.
    """
    if monthly_earnings <= 0 or max_monthly_benefit <= 0:
        raise ValueError("monthly_earnings and max_monthly_benefit must be positive")
    if not 0 < benefit_percent <= 100 or not 0 <= min_percent_of_benefit <= 100:
        raise ValueError("percentages must be between 0 and 100")
    if other_income_monthly < 0 or min_monthly_benefit < 0:
        raise ValueError("other income and minimum benefit cannot be negative")

    gross = monthly_earnings * benefit_percent / 100
    capped = min(gross, max_monthly_benefit)
    minimum = max(min_monthly_benefit, capped * min_percent_of_benefit / 100)
    net = max(capped - other_income_monthly, minimum)
    steps = [
        f"{benefit_percent:g}% x ${monthly_earnings:,.2f} = ${gross:,.2f}",
        f"capped at ${max_monthly_benefit:,.2f} -> ${capped:,.2f}",
        f"minus other income ${other_income_monthly:,.2f} -> ${capped - other_income_monthly:,.2f}",
        f"minimum benefit ${minimum:,.2f} -> estimated net ${net:,.2f}",
    ]
    return BenefitEstimate(round(gross, 2), round(capped, 2), other_income_monthly, round(minimum, 2), round(net, 2), steps)
