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


def configured_keys() -> list[str]:
    """Keys from .env, comma-separated: one per person, so access can be revoked individually."""
    return [key.strip() for key in os.getenv(API_KEY_ENV, "").split(",") if key.strip()]


def require_api_key(x_api_key: str | None = Security(api_key_header)) -> str:
    """Accept only requests carrying one of the configured keys in an X-API-Key header."""
    keys = configured_keys()  # read per request, so adding or revoking a key needs no code change
    if not keys:  # fail closed: a missing key must never mean "open to everyone"
        raise HTTPException(503, f"Server misconfigured: set {API_KEY_ENV} in .env")
    if x_api_key is None:
        raise HTTPException(401, "Missing or invalid X-API-Key header")
    # Constant-time compare against every key (no early exit), so timing reveals nothing.
    matches = [secrets.compare_digest(x_api_key, key) for key in keys]
    if not any(matches):
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
