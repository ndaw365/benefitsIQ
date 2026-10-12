"""API key check and a per-key rate limit, used as FastAPI dependencies."""

import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_ENV = "BENEFITSIQ_API_KEY"
# A declared security scheme (not a plain Header) gives /docs its "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Security(api_key_header)) -> str:
    """Accept only requests carrying the key from .env in an X-API-Key header."""
    expected = os.getenv(API_KEY_ENV)  # read per request, so a key change needs no code change
    if not expected:  # fail closed: a missing key must never mean "open to everyone"
        raise HTTPException(503, f"Server misconfigured: set {API_KEY_ENV} in .env")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):  # constant-time compare
        raise HTTPException(401, "Missing or invalid X-API-Key header")
    return x_api_key


class RateLimiter:
    """Sliding window: at most `limit` cost units per `window` seconds per API key.

    In memory and per process: it resets on restart and isn't shared across workers.
    Production would use a shared store (e.g. Redis) or the API gateway's limiter.
    """

    def __init__(self, limit: int = 20, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def check(self, key: str, cost: int = 1) -> None:
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0][0] >= self.window:  # forget requests older than the window
            hits.popleft()
        if sum(c for _, c in hits) + cost > self.limit:
            retry_after = int(self.window - (now - hits[0][0])) + 1 if hits else int(self.window)
            raise HTTPException(429, "Rate limit exceeded for this API key", headers={"Retry-After": str(retry_after)})
        hits.append((now, cost))
