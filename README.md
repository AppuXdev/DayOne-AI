# DayOne AI

DayOne AI is a multi-tenant HR knowledge copilot designed like a SaaS product: secure organization boundaries, admin controls, real-time chat, and continuous content updates.

It helps teams answer policy, benefits, and onboarding questions in seconds, grounded in each tenant's own documents.

## Product Snapshot

- Multi-tenant architecture with tenant-isolated data and indexes.
- Hybrid retrieval stack (FAISS + BM25 + RRF) with optional reranking.
- Real-time streaming chat via Server-Sent Events (SSE).
- Admin workspace for uploads, user administration, and drift reporting.
- Feedback loop that improves retrieval quality over time.

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
