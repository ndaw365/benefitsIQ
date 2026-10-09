"""Tests for ask() using a fake client, so no API key or network is needed."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types
from tenacity import wait_none

from src import llm
from src.schemas import PolicyAnswer

VALID_JSON = json.dumps(
    {"answer": "90 days.", "confidence": "high", "sources": ["p.12"], "needs_human_review": False}
)


def fake_response(text: str | None, finish: types.FinishReason = types.FinishReason.STOP):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=0
        ),
    )


class FakeClient:
    """Returns (or raises) the queued items in order, one per API call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def use_client(monkeypatch, tmp_path):
    """Install a fake client, log to a temp file, and skip retry sleeps."""
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(llm._call_model.retry, "wait", wait_none())

    def install(*outcomes) -> FakeClient:
        client = FakeClient(*outcomes)
        monkeypatch.setattr(llm, "get_client", lambda: client)
        return client

    return install


def read_log() -> list[dict]:
    return [json.loads(line) for line in llm.LOG_PATH.read_text().splitlines()]


def test_plain_text(use_client) -> None:
    use_client(fake_response("hello"))
    assert llm.ask("hi") == "hello"


def test_schema_parses_into_model(use_client) -> None:
    use_client(fake_response(VALID_JSON))
    result = llm.ask("q", schema=PolicyAnswer)
    assert isinstance(result, PolicyAnswer)
    assert result.sources == ["p.12"]


def test_truncation_raises(use_client) -> None:
    use_client(fake_response('{"answer": "90 da', types.FinishReason.MAX_TOKENS))
    with pytest.raises(llm.TruncatedOutputError):
        llm.ask("q", schema=PolicyAnswer)


def test_empty_text_raises(use_client) -> None:
    use_client(fake_response(None, types.FinishReason.SAFETY))
    with pytest.raises(llm.EmptyResponseError):
        llm.ask("q")


def test_schema_mismatch_raises(use_client) -> None:
    use_client(fake_response('{"answer": "x", "confidence": "very high"}'))
    with pytest.raises(llm.MalformedOutputError):
        llm.ask("q", schema=PolicyAnswer)


def test_retries_rate_limit_then_succeeds(use_client) -> None:
    client = use_client(errors.ClientError(429, {}), fake_response("ok"))
    assert llm.ask("q") == "ok"
    assert client.calls == 2
    log = read_log()
    assert "ClientError" in log[0]["error"] and log[1]["error"] is None


def test_does_not_retry_bad_request(use_client) -> None:
    client = use_client(errors.ClientError(400, {}), fake_response("never used"))
    with pytest.raises(errors.ClientError):
        llm.ask("q")
    assert client.calls == 1


def test_log_has_tokens_and_latency(use_client) -> None:
    use_client(fake_response("hello"))
    llm.ask("abc")
    (entry,) = read_log()
    assert entry["prompt_chars"] == 3
    assert entry["tokens_in"] == 10 and entry["tokens_out"] == 5
    assert entry["finish_reason"] == "STOP"
    assert entry["latency_ms"] >= 0
