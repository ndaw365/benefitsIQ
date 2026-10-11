"""Benefit arithmetic: cap, offsets, and the minimum floor."""

import pytest

from src.agent.benefit import calculate_benefit


def test_simple_percent():
    assert calculate_benefit(5400, 60, 10_000).net == 3240.0


def test_cap_applies():
    est = calculate_benefit(20_000, 60, 10_000)
    assert (est.gross, est.capped, est.net) == (12_000.0, 10_000.0, 10_000.0)


def test_offset_then_minimum_floor():
    # claim_05: $3,100 x 60% = $1,860; minus SSDI $1,850 = $10; floor = max($100, 10% of $1,860) = $186
    est = calculate_benefit(3100, 60, 10_000, other_income_monthly=1850, min_monthly_benefit=100, min_percent_of_benefit=10)
    assert (est.capped, est.minimum, est.net) == (1860.0, 186.0, 186.0)
    assert len(est.steps) == 4


@pytest.mark.parametrize("kwargs", [
    dict(monthly_earnings=0, benefit_percent=60, max_monthly_benefit=10_000),
    dict(monthly_earnings=5000, benefit_percent=160, max_monthly_benefit=10_000),
    dict(monthly_earnings=5000, benefit_percent=60, max_monthly_benefit=10_000, other_income_monthly=-5),
])
def test_rejects_impossible_inputs(kwargs):
    with pytest.raises(ValueError):
        calculate_benefit(**kwargs)
