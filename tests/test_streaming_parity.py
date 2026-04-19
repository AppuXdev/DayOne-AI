from __future__ import annotations

import json

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if not block.startswith("data: "):
            continue
        payload = block[len("data: ") :]
        events.append(json.loads(payload))
    return events


def test_stream_uses_shared_processing_pipeline(monkeypatch) -> None:
    response = main.ChatResponse(
        answer="Final grounded answer.",
        sources=[],
        username="john_doe",
        organization="org_acme",
        role="admin",
        model="test-model",
        confidence=0.81,
        confidence_label="high",
        conflict_detected=False,
        latency_ms=245.1,
        ttft_ms=88.4,
        verification={
            "is_grounded": True,
            "verification_confidence": 0.9,
            "unsupported_claims": [],
            "conflict_detected": False,
        },
        abstained=False,
        abstain_reason=None,
        route="expanded",
        query_type="ambiguous",
        query_id="qid-123",
    )

    monkeypatch.setattr(
        main,
        "decode_token",
        lambda _token: main.TokenPayload(
            sub="john_doe",
            username="john_doe",
            organization="org_acme",
            tenant_id="tenant-1",
            role="admin",
        ),
        raising=True,
    )
    monkeypatch.setattr(
        main,
        "process_chat_query",
        lambda _payload, _claims: main.ProcessedQueryResult(
            response=response,
            buffered_tokens=["Final ", "grounded ", "answer."],
        ),
        raising=True,
    )

    client = TestClient(main.app)
    res = client.post("/api/chat/stream", json={"prompt": "What is PTO?", "token": "token"})

    assert res.status_code == 200
    events = _parse_sse_events(res.text)
    assert events[0]["type"] == "meta"
    assert events[0]["route"] == "expanded"
    assert events[0]["query_type"] == "ambiguous"
    assert events[0]["verification"]["is_grounded"] is True

    token_events = [e for e in events if e.get("type") == "token"]
    assert [e["content"] for e in token_events] == ["Final ", "grounded ", "answer."]

    done = [e for e in events if e.get("type") == "done"][0]
    assert done["query_id"] == "qid-123"
    assert done["abstained"] is False


def test_stream_emits_error_event_on_processing_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "decode_token",
        lambda _token: main.TokenPayload(
            sub="john_doe",
            username="john_doe",
            organization="org_acme",
            tenant_id="tenant-1",
            role="admin",
        ),
        raising=True,
    )

    def _raise(_payload, _claims):
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    monkeypatch.setattr(main, "process_chat_query", _raise, raising=True)

    client = TestClient(main.app)
    res = client.post("/api/chat/stream", json={"prompt": "trigger", "token": "token"})

    assert res.status_code == 200
    events = _parse_sse_events(res.text)
    assert events[0]["type"] == "error"
    assert "Prompt cannot be empty" in events[0]["detail"]
