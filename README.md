# DayOne AI

DayOne AI is a multi-tenant HR knowledge copilot designed like a SaaS product: secure organization boundaries, admin controls, real-time chat, and continuous content updates.

It helps teams answer policy, benefits, and onboarding questions in seconds, grounded in each tenant's own documents.

## Product Snapshot

- Multi-tenant architecture with tenant-isolated data and indexes.
- Hybrid retrieval stack (FAISS + BM25 + RRF) with optional reranking.
- Real-time streaming chat via Server-Sent Events (SSE).
- Admin workspace for uploads, user administration, and drift reporting.
- Feedback loop that improves retrieval quality over time.

## Evaluation-First Approach

DayOne AI treats evaluation as a release gate, not a side task.

Tier 1 (default, deterministic):

- Retrieval hit rate
- Correct abstentions for negative queries
- Precision@1 / @3 / @k (keyword-in-chunk proxy)
- End-to-end latency and TTFT (time-to-first-token)
- Confidence tracking and error category breakdown

Tier 2 (optional judge mode):

- Faithfulness: are claims grounded in retrieved context?
- Correctness: does the answer satisfy the question intent?
- Hallucination flag per query
- Cache-backed judge runs for affordable repeated evaluation

Commands:

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme
.\.venv\Scripts\python.exe eval.py --org org_acme --judge
.\.venv\Scripts\python.exe eval.py --org org_acme --judge --rerun
```

## Why This Feels Like SaaS

- Tenant isolation by design:
  - Data layout and vector indexes are separated per organization.
- Role-based experiences:
  - Employee chat and admin workflows are distinct.
- Operational tooling:
  - Auto-ingest watcher, health endpoint, evaluation harness, and test suite.
- API-first foundation:
  - FastAPI backend powers both Streamlit and Next.js clients.

## Key Features

### Employee Experience

- Ask natural-language HR questions and receive grounded answers.
- View confidence signals and source-backed rationale.
- Submit feedback on response quality.

### Admin Experience

- Upload tenant documents and trigger index refresh.
- Manage tenant users (create, update, delete).
- Review semantic drift trends for governance and quality checks.

### Platform Capabilities

- Retrieval pipeline: FAISS dense retrieval + BM25 sparse retrieval + RRF fusion.
- Optional cross-encoder reranker (`DAYONE_USE_RERANKER=1` by default).
- Streaming endpoint (`/api/chat/stream`) with TTFT event support.
- Tenant-level limits and configurable auth/session controls.

## Feedback Learning Mechanism

Feedback is not a generic log. It changes ranking weights used by retrieval.

Implementation summary:

- Feedback is logged per organization with rating (`up` / `down`) and cited source files.
- Per-source stats are converted into a bounded reputation score.
- Retrieval multiplies base candidate score by source weight before final ranking.

Formula used in [feedback.py](feedback.py):

$$
  \\text{net\_ratio} = \\frac{\\text{positive} - \\text{negative}}{\\max(\\text{total}, 1)}
$$

$$
  \\text{reputation} = 0.5 + 0.5 \\cdot \\tanh(2 \\cdot \\text{net\_ratio})
$$

$$
  \\text{weight} = 1.0 + (\\text{reputation} - 0.5)\\ \\in\\ [0.77, 1.23]
$$

Applied in [retriever.py](retriever.py):

$$
  \\text{final\_score} = \\text{base\_score} \\times \\text{source\_weight}
$$

Where `base_score` is:

- Cross-encoder score when reranker is ON
- BM25-derived score when reranker is OFF

Why bounded weighting matters:

- A single bad vote cannot erase a source.
- A single good vote cannot dominate retrieval.
- The model adapts over time while staying stable.

## Drift Definition and Measurement

Drift is defined as semantic change between old vs new document chunks, not as a vague trend line.

Current method in [drift.py](drift.py):

1. Embed old and new chunk sets.
2. L2-normalize embeddings and compute cosine distance.
3. For each new chunk, find nearest old chunk.
4. Mark as `changed` if distance exceeds threshold (`DRIFT_DISTANCE_THRESHOLD`, default `0.25`).
5. Detect `removed` and net `new` sections from unmatched counts.

Report output includes:

- `changed_chunks`
- `unchanged_chunks`
- `removed_chunks`
- `new_chunks`
- Per-diff snippet pairs with measured distance

Operational meaning:

- Drift here is embedding-level policy-content shift at chunk granularity.
- This is document semantic versioning, not dashboard buzzwording.

## Retrieval Optimization Insights

Key engineering trade-offs from [retriever.py](retriever.py):

- Hybrid retrieval over dense-only:
  - Dense retrieval misses exact lexical signals (codes, policy terms, proper nouns).
  - BM25 recovers lexical precision.
  - RRF fuses both without fragile score normalization.
- Reranker ON by default:
  - Typical gain: ~10-16 percentage points precision uplift (project benchmark note).
  - Typical cost: ~200-400 ms extra CPU latency.
- Candidate set tuning:
  - `CANDIDATE_K=12` balances recall and reranker compute.
  - `FINAL_K=4` constrains context passed to the LLM.

## Hard ML Problem This System Solves

The core challenge is retrieval under ambiguity with strict grounding constraints.

In practice, HR queries are often:

- Lexically vague ("leave policy")
- Semantically specific (carry-forward, eligibility windows)
- Risk-sensitive (wrong answer has policy/compliance impact)

The system addresses this by combining:

- Dual-signal retrieval (semantic + lexical)
- Rank fusion and cross-encoder reranking
- Confidence and abstention behavior for low-evidence cases
- Online source-weight adaptation from user feedback

This is not model fine-tuning; it is production retrieval intelligence engineered for stability, explainability, and tenant safety.

## Failure Case Example (And How We Debug It)

Example query:

- "Do unused leaves expire?"

Observed failure pattern (pre-hardening runs):

- Retrieved chunk emphasized generic leave policy language.
- Correct carry-forward section ranked lower.
- Answer reflected the higher-ranked but less specific chunk.

Root cause:

- Lexical overlap around "leave" outweighed the more specific "unused/carry-forward" intent in top candidates.

Mitigations now used:

- Hybrid retrieval (dense + BM25 + RRF) to reduce single-mode bias.
- Cross-encoder reranking to promote semantically aligned chunks.
- Source feedback weighting to down-rank repeatedly unhelpful sources.

Next hardening step (planned):

- Query rewriting for ambiguity resolution before retrieval.

## Architecture

### Frontends

- Streamlit app ([app.py](app.py)):
  - Employee chat + admin portal in one UI.
- Next.js app ([app/page.tsx](app/page.tsx)):
  - Routes for landing, login, chat, and admin dashboards.

### Backend

- FastAPI API server ([main.py](main.py)).
- Auth: JWT-based API auth plus Streamlit authenticator support.
- Core services:
  - Retrieval ([retriever.py](retriever.py))
  - Ingestion ([ingest.py](ingest.py))
  - Auto-ingest watcher ([auto_ingest.py](auto_ingest.py))
  - Feedback weighting ([feedback.py](feedback.py))
  - Drift analysis ([drift.py](drift.py))

## API Overview

Auth:

- `POST /auth/login`

Chat and feedback:

- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/feedback`

Admin endpoints:

- `POST /api/admin/upload`
- `GET /api/admin/drift-report`
- `GET /api/admin/users`
- `POST /api/admin/users`
- `PATCH /api/admin/users/{username}`
- `DELETE /api/admin/users/{username}`

Operations:

- `GET /health`

## Multi-Tenant Data Model

- Source content: [data/org_acme](data/org_acme), [data/org_globex](data/org_globex), and other `data/org_<tenant>` folders.
- Per-tenant vector index: `vector_store/org_<tenant>/index.faiss`.
- Cache manifests and partitioned parts under `vector_cache/org_<tenant>/`.
- Watcher monitors tenant folders for `.pdf` and `.csv` changes.

## Tech Stack

- Backend: Python, FastAPI, LangChain, FAISS, sentence-transformers.
- LLM provider: Groq (`llama-3.1-8b-instant` by default via env).
- Frontend: Next.js 14, React 18, TypeScript.
- Streamlit auth: `streamlit-authenticator`.

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- A valid `GROQ_API_KEY`

Create `.env` in project root:

```env
GROQ_API_KEY=your_key_here
# Optional overrides
# DAYONE_GROQ_MODEL=llama-3.1-8b-instant
# DAYONE_USE_RERANKER=1
# JWT_SECRET_KEY=replace_in_production
# CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### One-Command Bootstrap (Recommended)

```powershell
.\run.ps1
```

What this does:

- Creates `.venv` if needed.
- Installs Python and Node dependencies.
- Runs ingestion.
- Starts auto-ingest watcher, FastAPI, and Next.js.
- Starts Streamlit unless `-NoStreamlit` is provided.

Useful flags:

```powershell
.\run.ps1 -SkipInstall
.\run.ps1 -SkipIngest
.\run.ps1 -NoStreamlit
.\run.ps1 -UseSeparateTerminals
```

### Manual Startup

```powershell
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe auto_ingest.py
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
npm install
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'
npm run dev
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Demo Credentials (Local Only)

- Employee: `john_doe` / `password123`
- Admin: `admin_acme` / `password123`

Rotate all secrets and credentials before staging or production use.

## Quality and Evaluation

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Latest recorded status (April 2026): `8 passed`.

Run retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme
.\.venv\Scripts\python.exe eval.py --org org_acme --output eval_results.json
.\.venv\Scripts\python.exe eval.py --org org_acme --judge
```

## Configuration Reference

- `GROQ_API_KEY` (required)
- `DAYONE_GROQ_MODEL` (optional)
- `DAYONE_USE_RERANKER` (`1` or `0`)
- `TENANT_RATE_LIMIT_RPM`
- `TENANT_UPLOAD_LIMIT_PER_DAY`
- `TENANT_MAX_CHUNKS`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `JWT_SECRET_KEY`
- `CORS_ORIGINS`

## Troubleshooting

- Missing Groq API key:
  - Add `GROQ_API_KEY` to `.env` and restart services.
- Empty tenant knowledge base:
  - Run ingestion and confirm docs exist under `data/org_*`.
- Next.js cannot reach API:
  - Confirm `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`.
- Watcher not triggering:
  - Ensure [auto_ingest.py](auto_ingest.py) is running and files changed under tenant folders.

## Production Readiness Note

This repository is structured for local development, product prototyping, and internal demos. For production, enforce stronger secrets management, tenant hardening, observability, and deployment controls.
