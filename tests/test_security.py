"""Rate limiter behavior with a fake clock, so time is controlled exactly."""

import pytest
from fastapi import HTTPException

from src.api.security import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(limit=3, window=60, clock=FakeClock())
    for _ in range(3):
        limiter.check("k")
    with pytest.raises(HTTPException) as exc:
        limiter.check("k")
    assert exc.value.status_code == 429 and exc.value.headers["Retry-After"] == "61"


def test_window_slides():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.check("k")
    clock.now += 60
    limiter.check("k")  # the first request has aged out


def test_cost_counts_against_budget_and_keys_are_separate():
    limiter = RateLimiter(limit=5, window=60, clock=FakeClock())
    limiter.check("a", cost=5)
    with pytest.raises(HTTPException):
        limiter.check("a", cost=1)
    limiter.check("b", cost=5)  # another key has its own budget
