from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain.schema import Document

import main
from backend.services.abstention import AbstentionDecision
from backend.services.verifier import VerificationResult
from retriever import RetrievalResult


class _FakeChatMemory:
    def add_user_message(self, _message: str) -> None:
        pass

    def add_ai_message(self, _message: str) -> None:
        pass


class _FakeMemory:
    def __init__(self) -> None:
        self.chat_memory = _FakeChatMemory()


class _FakeLLM:
    def stream(self, _messages):
        yield SimpleNamespace(content="PTO is paid time off.")

    def invoke(self, _messages):
        return SimpleNamespace(content="PTO is paid time off.")


class _FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def retrieve(self, _query: str, candidate_k: int = 0):
        return replace(self._result)


def test_build_query_trace_payload_includes_runtime_stages() -> None:
    docs = [Document(page_content="PTO is paid time off.", metadata={"source": "benefits.csv"})]
    result = RetrievalResult(
        final_docs=docs,
        final_scores=[0.91],
        confidence=0.82,
        candidates=docs,
        candidate_scores=[0.74],
        latency_ms=12.3,
        used_reranker=True,
        dense_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.88}],
        sparse_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.73}],
        fused_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.61}],
        reranked_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.91, "raw_score": 0.84}],
    )
    source_metadata = [main.SourceMetadata(source="benefits.csv", metadata={"source_path": "benefits.csv"})]
    verification = VerificationResult(
        is_grounded=True,
        verification_confidence=0.93,
        unsupported_claims=[],
        conflict_detected=False,
    )

    payload = main.build_query_trace_payload(
        query="What is PTO?",
        tenant_id="tenant-1",
        query_type="factual",
        route="fast_path",
        result=result,
        final_sources=source_metadata,
        verification_result=verification,
        confidence=0.82,
        abstained=False,
        abstain_reason=None,
        latency_ms=123.4,
    )

    assert payload["tenant_id"] == "tenant-1"
    assert payload["retrieval"]["dense_topk"][0]["source"] == "benefits.csv"
    assert payload["verification"]["is_grounded"] is True
    assert payload["abstained"] is False
    assert payload["final_context"][0]["source"] == "benefits.csv"


def test_chat_writes_query_trace(monkeypatch) -> None:
    docs = [Document(page_content="PTO is paid time off.", metadata={"source": "benefits.csv"})]
    retrieval_result = RetrievalResult(
        final_docs=docs,
        final_scores=[0.91],
        confidence=0.82,
        candidates=docs,
        candidate_scores=[0.74],
        latency_ms=12.3,
        used_reranker=True,
        dense_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.88}],
        sparse_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.73}],
        fused_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.61}],
        reranked_topk=[{"rank": 1, "source": "benefits.csv", "score": 0.91, "raw_score": 0.84}],
    )

    monkeypatch.setattr(main, "decode_token", lambda _token: main.TokenPayload(sub="john_doe", username="john_doe", organization="org_acme", tenant_id="tenant-1", role="admin"), raising=True)
    monkeypatch.setattr(main, "load_embeddings", lambda: SimpleNamespace(), raising=True)
    monkeypatch.setattr(main, "get_or_create_memory", lambda *_args, **_kwargs: _FakeMemory(), raising=True)
    monkeypatch.setattr(main, "rewrite_query", lambda query, _memory, _llm: query, raising=True)
    monkeypatch.setattr(main, "get_feedback_store", lambda: SimpleNamespace(get_source_weights=lambda _org: {}), raising=True)
    monkeypatch.setattr(main, "build_hybrid_retriever", lambda *args, **kwargs: _FakeRetriever(retrieval_result), raising=True)
    monkeypatch.setattr(main, "ChatGroq", lambda **_kwargs: _FakeLLM(), raising=True)
    monkeypatch.setattr(main, "get_answer_verifier", lambda: SimpleNamespace(verify=lambda *_args, **_kwargs: VerificationResult(is_grounded=True, verification_confidence=0.93, unsupported_claims=[], conflict_detected=False)), raising=True)
    monkeypatch.setattr(main, "should_abstain", lambda **_kwargs: AbstentionDecision(abstained=False, reason=None), raising=True)
    monkeypatch.setattr(main.query_trace_db, "resolve_tenant_id", lambda **_kwargs: "tenant-1", raising=True)
    store_mock = MagicMock(return_value="trace-1")
    monkeypatch.setattr(main.query_trace_db, "store_query_trace", store_mock, raising=True)

    client = TestClient(main.app)
    response = client.post("/api/chat", json={"prompt": "What is PTO?", "token": "token"})

    assert response.status_code == 200
    assert store_mock.called
    _, kwargs = store_mock.call_args
    assert kwargs["tenant_id"] == "tenant-1"
    assert kwargs["trace"]["retrieval"]["rrf_fused"][0]["source"] == "benefits.csv"
    assert kwargs["trace"]["verification"]["is_grounded"] is True


def test_admin_traces_endpoint_filters(monkeypatch) -> None:
    sample_trace = {
        "id": "trace-1",
        "tenant_id": "tenant-1",
        "query": "What is PTO?",
        "trace": {"query_type": "factual", "abstained": False, "confidence": 0.82},
        "created_at": "2026-04-19T00:00:00Z",
    }
    monkeypatch.setattr(main.query_trace_db, "resolve_tenant_id", lambda **_kwargs: "tenant-1", raising=True)
    monkeypatch.setattr(main.auth_db, "is_enabled", lambda: True, raising=True)
    list_mock = MagicMock(return_value=[sample_trace])
    monkeypatch.setattr(main.query_trace_db, "list_query_traces", list_mock, raising=True)

    current_user = main.TokenPayload(sub="john_doe", username="john_doe", organization="org_acme", tenant_id="tenant-1", role="admin")
    traces = main.list_admin_traces(tenant_id="tenant-1", limit=10, query_type="factual", abstained=False, low_confidence=False, current_user=current_user)

    assert len(traces) == 1
    assert traces[0].id == "trace-1"
    assert list_mock.called
    _, kwargs = list_mock.call_args
    assert kwargs["tenant_id"] == "tenant-1"
    assert kwargs["query_type"] == "factual"
    assert kwargs["abstained"] is False
