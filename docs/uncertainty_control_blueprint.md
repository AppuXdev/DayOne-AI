# Uncertainty-Control Upgrade Blueprint

This document turns the roadmap into implementation-ready work.

## Goal

Upgrade DayOne AI from a strong retrieval pipeline to a retrieval system with explicit uncertainty control.

## Phase Plan

### Phase 1: Verification + Abstention

Scope:
- Add answer verification contract and abstention policy.
- Keep current chat pipeline behavior intact behind feature flags.

Files:
- `backend/services/verifier.py`
- `backend/services/abstention.py`

Acceptance criteria:
- Structured verification payload exists and is serializable.
- Abstention policy deterministic for defined thresholds.
- Policy unit tests pass.

### Phase 2: Query Classification + Routing

Scope:
- Add query classifier contract and route selection contracts.
- Start with simple, bounded route taxonomy.

Files:
- `backend/services/query_classifier.py`
- `backend/services/query_router.py`

Acceptance criteria:
- Query type contract includes: factual, ambiguous, multi_hop, exception.
- Route config is explicit and inspectable.
- Router behavior is deterministic.

### Phase 3: Retrieval Observability

Scope:
- Log pre-rerank, post-rerank, final selected traces per query.

Proposed table:
- `query_traces(id, tenant_id, query, route, trace_json, created_at)`

Acceptance criteria:
- Trace payload captures top-k before/after rerank.
- Trace includes route and latency.

### Phase 4: Feedback Learning (Bounded)

Scope:
- Move from source-level feedback to query-aware bounded adjustment.

Proposed table:
- `feedback_signals(id, tenant_id, query_embedding, chunk_id, signal, created_at)`

Acceptance criteria:
- Influence capped.
- Decay applied over time.
- No unbounded score drift.

### Phase 5: Continuous Evaluation

Scope:
- Collect failures from production behavior and fold into rolling eval set.

Data:
- `eval_set/curated.json`
- `eval_set/recent_failures.json`

Acceptance criteria:
- Regressions evaluated against rolling baseline.
- Gate criteria explicit and versioned.

## Critical Rules

1. Everything observable.
2. Everything bounded.
3. No silent decisions.
4. Modules remain isolated.

## Suggested PR Sequence

1. PR-1: Contracts only (`verifier`, `abstention`, `classifier`, `router`) + tests.
2. PR-2: Integrate verifier + abstention in API response path behind flags.
3. PR-3: Add retrieval trace logging and admin debug read endpoint.
4. PR-4: Add bounded feedback-learning adjustments.
5. PR-5: Add continuous eval ingestion and release gate rules.

## Environment Flags (proposed)

- `DAYONE_ENABLE_VERIFIER=0|1`
- `DAYONE_ENABLE_ABSTENTION=0|1`
- `DAYONE_ABSTAIN_RETRIEVAL_THRESHOLD=0.40`
- `DAYONE_ENABLE_QUERY_ROUTING=0|1`
- `DAYONE_ENABLE_TRACE_LOGGING=0|1`

## Near-Term KPI Targets

- Lower unsupported-claim rate.
- Higher abstention precision.
- Stable latency p95 under routing.
- Faster RCA from trace visibility.
