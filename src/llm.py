"""Single gateway for every LLM call: retries, logging, structured output, truncation checks."""

import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, overload

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # confirm in AI Studio; names change
LOG_PATH = Path("logs/llm_calls.jsonl")
RETRYABLE_CODES = {429, 500, 503, 504}  # rate limit + transient server errors

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for model output we refuse to pass downstream."""


class TruncatedOutputError(LLMError):
    """The model hit max_output_tokens before finishing."""


class EmptyResponseError(LLMError):
    """The model returned no text (e.g. blocked by a safety filter)."""


class MalformedOutputError(LLMError):
    """The model's JSON did not match the requested schema."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the API client on first use (reads GEMINI_API_KEY from the environment)."""
    return genai.Client()


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in RETRYABLE_CODES


def _log_call(record: dict) -> None:
    """Append one JSON object per line, so a crash never corrupts earlier entries."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_model(
    prompt: str, config: types.GenerateContentConfig
) -> types.GenerateContentResponse:
    """One API attempt. Every attempt is logged, including failed ones."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompt_chars": len(prompt),
        "schema": config.response_schema.__name__ if config.response_schema else None,
    }
    start = time.perf_counter()
    try:
        response = get_client().models.generate_content(
            model=MODEL, contents=prompt, config=config
        )
    except Exception as exc:
        record.update(
            latency_ms=round((time.perf_counter() - start) * 1000),
            error=f"{type(exc).__name__}: {exc}"[:300],
        )
        _log_call(record)
        raise

    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    record.update(
        latency_ms=round((time.perf_counter() - start) * 1000),
        tokens_in=usage.prompt_token_count if usage else None,
        tokens_out=usage.candidates_token_count if usage else None,
        tokens_thinking=usage.thoughts_token_count if usage else None,
        finish_reason=candidate.finish_reason.name if candidate and candidate.finish_reason else None,
        error=None,
    )
    _log_call(record)
    return response


@overload
def ask(prompt: str, *, system: str | None = ..., schema: None = ...,
        temperature: float = ..., max_tokens: int = ...) -> str: ...
@overload
def ask(prompt: str, *, system: str | None = ..., schema: type[T],
        temperature: float = ..., max_tokens: int = ...) -> T: ...


def ask(
    prompt: str,
    *,
    system: str | None = None,
    schema: type[T] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str | T:
    """Send a prompt to Gemini.

    Without a schema, returns the text. With a Pydantic schema, the model is
    constrained to that JSON shape and a validated instance is returned.
    Token counts and latency go to logs/llm_calls.jsonl, not the return value.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    response = _call_model(prompt, config)

    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.finish_reason == types.FinishReason.MAX_TOKENS:
        raise TruncatedOutputError(f"Output cut off at max_tokens={max_tokens}")
    if not response.text:
        reason = candidate.finish_reason.name if candidate and candidate.finish_reason else "unknown"
        raise EmptyResponseError(f"Empty response (finish_reason={reason})")

    if schema is None:
        return response.text
    try:
        return schema.model_validate_json(response.text)
    except ValidationError as exc:
        raise MalformedOutputError(f"Output did not match {schema.__name__}: {exc}") from exc
