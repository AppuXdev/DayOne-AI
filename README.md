# DayOne AI

Production-grade, multi-tenant retrieval system with measurable guarantees.

Onboarding should not require guesswork, especially when policies are ambiguous.
DayOne AI is built to answer those questions with grounded, auditable responses.

It is built around a simple constraint:

**Answers must be explainable, tenant-isolated, and measurable — not just plausible.**

Unlike typical RAG demos, this system treats retrieval quality, data consistency, and operational safety as enforceable properties.

It combines hybrid retrieval, cross-encoder reranking, feedback-weighted ranking, and lifecycle-aware ingestion into a single, database-native system.

The system was migrated from FAISS to PostgreSQL pgvector only after explicit parity validation, ensuring no loss in retrieval quality.

## Overview

DayOne AI is a multi-tenant knowledge assistant platform for organizations that need reliable policy answers under real-world ambiguity.

It includes:

- FastAPI backend for auth, retrieval orchestration, ingestion, admin APIs, and streaming
- Next.js frontend for SaaS-style product UX and admin workflows
- PostgreSQL plus pgvector as canonical dense retrieval backend
- MinIO for document object storage and lifecycle consistency checks
- In-process conversation memory (Redis is provisioned in infra; persistence migration path is prepared)
- Streamlit interface for fast internal ops and live streaming interaction

## 🚀 What Makes This Different

This system is not a "chat with your documents" demo.

It is built around four enforceable properties:

- **Evaluation-first design**  
  Retrieval quality is measured with explicit benchmarks, not assumed.

- **Explainable retrieval**  
  Every answer includes justification traces and source attribution.

- **Feedback as a ranking signal**  
  User feedback directly influences retrieval ordering via bounded weighting.

- **Strict multi-tenant isolation**  
  Every query, embedding, and document operation is scoped by tenant_id.

These constraints are not edge cases.

**They are the system.**

## 🧠 Problem

The core challenge is simple:

**Answer ambiguous questions with strictly grounded, tenant-safe information.**

Naive RAG systems fail for a simple reason:

**They optimize for plausibility instead of correctness.**

- Dense-only retrieval misses exact policy terms
- Sparse-only retrieval misses semantic intent
- No reranking produces unstable top-k results
- No evaluation turns quality into guesswork
- No lifecycle control creates inconsistent data states

DayOne AI treats these as system constraints, not edge cases.

## Key Features

- Multi-tenant enforcement at every layer
- DB-only auth and user management (`DATABASE_URL` required at runtime)
- Hybrid retrieval: BM25 + dense retrieval + Reciprocal Rank Fusion
- Cross-encoder reranking for precision-critical answers
- Explainable retrieval with justification records and source transparency
- Feedback-weighted retrieval that updates source influence over time
- Drift detection for semantic policy change tracking
- Evaluation harness with benchmark metrics, parity validation, and stabilization gates
- Streaming responses over SSE and Streamlit for low-friction user interaction
- Lifecycle-aware ingestion with document status transitions and reconciliation

## ⚙️ Architecture

### System Overview

```mermaid
flowchart TD
  A[Frontend<br/>Next.js / Streamlit] --> B[FastAPI<br/>auth, retrieval orchestration, ingestion]
  B --> C[PostgreSQL<br/>users, documents, pgvector embeddings]
  B --> D[MinIO<br/>object storage]
  B --> E[Redis<br/>memory / cache]
```

All components are designed to be stateless or externally backed, enabling straightforward cloud deployment without architectural changes.

In practice, this means deployment can scale without redesigning core interfaces.

### High-level components

- API Layer (FastAPI)
  - Auth, chat APIs, admin APIs, upload and ingestion endpoints, feedback, streaming
  - Login requires `username`, `password`, and `organization`; JWT carries `tenant_id`
- Retrieval Layer
  - Candidate generation (sparse + dense), fusion, reranking, confidence estimation
  - Dense retrieval is pgvector-only and tenant-scoped by `tenant_id`
- Data Layer
  - PostgreSQL tables for tenants, users, documents, embeddings, metadata
  - pgvector indexes for ANN dense retrieval
  - MinIO for source objects
  - In-memory conversation store in current runtime
- Product Layer
  - Next.js app for employees and admins
  - Streamlit interface for rapid internal operations

### Query data flow

1. User query arrives with tenant identity from JWT context
2. Query is rewritten for retrieval robustness
3. Sparse and dense candidates are fetched under strict tenant filter
4. Reciprocal Rank Fusion combines sparse and dense rankings
5. Cross-encoder reranking refines top candidates
6. Context is assembled with source metadata and confidence
7. LLM generates grounded answer from retrieved context only
8. Response streams to client (SSE or Streamlit), with traceable sources

### Ingestion data flow

1. Admin uploads document
2. Object stored in MinIO under tenant-scoped key
3. Document row created with status lifecycle (uploading to processing to active or failed)
4. Chunking and embedding generation run per tenant snapshot
5. Embeddings are replaced transactionally in PostgreSQL
6. Reconciliation checks detect missing objects or orphaned metadata

Operational endpoint:

- `POST /api/admin/storage/reconcile` verifies and repairs object/DB consistency for the tenant scope

## 🔍 Retrieval Pipeline

### Why hybrid retrieval

DayOne AI uses hybrid retrieval because enterprise policy search has **two different needs**:

- lexical precision (exact policy terms, IDs, compliance phrases)
- semantic understanding (paraphrases, intent, ambiguity)

Retrieval combines both signals:

- BM25 captures exact policy terms, procedural phrases, and compliance tokens
- Dense retrieval captures paraphrase and semantic intent
- RRF unifies both without forcing incompatible score normalization assumptions

### Why reranking

Candidate retrieval maximizes recall.

Reranking maximizes final precision.

- Cross-encoder reranking improves top-answer quality when multiple near-relevant passages exist
- It adds latency, so it is treated as an explicit quality-latency tradeoff, not a hidden default

### Explainability and control

Retrieval outputs include candidate reasoning signals:

- Fusion and rerank effects
- Source attribution and confidence
- Justification traces for debugging and review

This is critical for production trust and incident response.

## 📊 Evaluation

**Evaluation is a system feature, not a notebook afterthought.**

No retrieval change is accepted without passing this evaluation layer.
All architectural changes (including FAISS -> pgvector migration) are gated by measured parity against this evaluation harness.

### Benchmark design

The benchmark set includes mixed query categories:

- Direct policy questions
- Paraphrases
- Multi-hop policy intent
- Negative queries requiring abstention
- Ambiguous prompts requiring grounded disambiguation

### Tracked metrics

- Positive retrieval hit rate
- Precision at 1, 3, and k
- Average latency and time-to-first-token
- Confidence behavior
- Error category distribution

### FAISS to pgvector parity validation

The dense backend migrated from FAISS to pgvector only after explicit parity checks.

| Setup | Hit Rate | P@1 | P@3 | P@k | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAISS baseline | 62.5% | 100.0% | 77.1% | 76.6% | 4550 ms |
| pgvector | 62.5% | 100.0% | 77.1% | 76.6% | 4665 ms |

Interpretation:

- Retrieval quality parity achieved
- Small latency increase is expected for database-backed ANN and accepted within gate thresholds

### Stabilization gate

A one-command gate validates post-migration safety using thresholded checks for:

- Hit rate delta
- P@1 delta
- Confidence delta
- Latency ratio

This prevents regressions from being merged on intuition.
The stabilization gate enforces release-level guarantees and prevents unverified retrieval changes from entering the system.

Gate implementation detail:

- The current gate compares `eval_pgvector.json` against immutable baseline in `scripts/legacy_benchmark/faiss_baseline_org_acme.json`

## 🧱 System Design Decisions

### PostgreSQL plus pgvector over standalone FAISS

Why this choice:

- Operationally simpler for multi-tenant SaaS data governance
- Better alignment with transaction boundaries and metadata consistency
- Easier backup, migration, and audit workflows than separate index artifact management

Tradeoff:

- Slight latency overhead versus in-process FAISS for equivalent quality

### Tenant isolation as a hard invariant

Isolation is enforced in:

- Auth token context
- Retrieval queries
- Embedding selection
- Document storage keys
- Admin operations

Goal: no cross-tenant context access even under ambiguous requests or operational failure.

**Isolation is a guarantee, not a best-effort behavior.**

### Feedback-weighted ranking

User feedback is not logged and ignored. It modifies source influence to improve retrieval ordering over time while preserving bounded stability.

### Lifecycle-aware ingestion

Documents are managed through explicit states to avoid silent inconsistency:

- uploading
- processing
- active
- failed
- deleted

MinIO and PostgreSQL reconciliation prevents drift between object and metadata layers.

## ⚠️ Failure Modes

Known limitations and active risk areas:

- Answer correctness is still partially estimated by LLM-as-judge; this is weaker than fully human-labeled gold datasets
- Reranker improves precision but can increase tail latency under heavy load
- Drift tracking currently emphasizes document semantic change more than live query-distribution drift
- Ambiguous policy questions can still underperform when source material is sparse or contradictory
- Operational reliability depends on healthy Postgres and MinIO boundaries; chat memory is currently process-local

These are explicit engineering constraints, not hidden caveats.

Example failure case:

Query: "Do unused leaves expire?"

Failure mode:

- Retrieval surfaced a carry-forward policy instead of expiration rules
- Caused by keyword overlap and insufficient query specificity

Mitigation:

- Query rewriting and hybrid retrieval improve recall
- Reranking prioritizes semantically aligned chunks

These limitations are actively monitored and prioritized over feature expansion.

## 🚀 Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- Groq API key

### Environment

Create a .env file with required runtime configuration:

```bash
GROQ_API_KEY=your_key
DATABASE_URL=postgresql+psycopg://dayone:dayone@localhost:5432/dayone
REDIS_URL=redis://localhost:6379/0
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password
MINIO_BUCKET=dayone-docs
DAYONE_USE_RERANKER=1
DAYONE_PGVECTOR_PROBES=10
DAYONE_EMBEDDING_DIM=384
```

Auth and users note:

- `DATABASE_URL` is mandatory for `/auth/login` and `/api/admin/users*`
- Legacy YAML users can be imported once with:

```powershell
.\.venv\Scripts\python.exe migrate_users.py
```

### Run infrastructure

```bash
docker compose -f infra/docker-compose.yml up --build
```

### Run evaluation

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme --output eval_pgvector.json
```

### Run stabilization gate

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stabilization_gate.ps1 -Org org_acme
```

## 🎬 Demo

This is designed to be shown, not just described.

The system is designed to be demonstrated end-to-end without hidden setup or manual intervention.

### Employee flow

- Authenticate to tenant-scoped account
- Ask ambiguous policy question
- Receive streamed answer with grounded source context
- Submit feedback to influence future ranking

### Admin flow

- Upload CSV or PDF policy files
- Trigger ingestion and monitor lifecycle status
- Validate drift report after document updates
- Run storage reconciliation endpoint to detect mismatches

### Suggested live demo script

1. Log in as tenant admin and upload revised policy document
2. Show ingestion status transition to active
3. Ask the same question before and after update
4. Show changed retrieval evidence and answer grounding
5. Run evaluation and stabilization gate to demonstrate measurable reliability

## Repository Structure

- app and frontend: product UX layers
- main.py: FastAPI API and orchestration
- retriever.py: hybrid retrieval, fusion, reranking, confidence
- ingest.py and auto_ingest.py: ingestion lifecycle and refresh
- eval.py: benchmark and metric pipeline
- backend/services: auth, user, document, embedding, and storage adapters
- scripts: DB init, stabilization gate, and ops utilities

## Why This Project Matters

DayOne AI demonstrates production ML systems engineering where retrieval quality, tenant safety, and operational correctness are treated as enforceable properties.

This is the difference between a RAG demo and a deployable multi-tenant AI system.

---

If your system cannot explain *why* it answered, it should not answer at all.
