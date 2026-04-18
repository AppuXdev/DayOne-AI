# DayOne AI

Multi-tenant HR onboarding assistant with retrieval-augmented generation (RAG), Streamlit and Next.js frontends, and a FastAPI backend.

## Latest Updates (April 2026)

- Added tenant-scoped admin user management in Streamlit and FastAPI.
- Added feedback-weighted retrieval via `/api/feedback` and `feedback.py`.
- Added SSE chat streaming endpoint `/api/chat/stream` with TTFT events.
- Added semantic drift reporting endpoint `/api/admin/drift-report`.
- Enabled auto-ingest watcher for `data/org_*/` changes.
- Stabilized Streamlit auth (`streamlit-authenticator`, `auto_hash=False`).
- Added sample PDF generation utility (`generate_sample_pdfs.py`).

## Core Capabilities

- Hybrid retrieval: FAISS dense + BM25 sparse + RRF fusion.
- Optional cross-encoder reranker (`DAYONE_USE_RERANKER=1` by default).
- Tenant-isolated document and index layout (`data/org_*`, `vector_store/org_*`).
- JWT auth and role-based API authorization.
- Admin upload flow with index rebuild and drift summary.
- Evaluation harness with deterministic Tier 1 and optional judge mode.

## Tech Stack

- Python: Streamlit, FastAPI, LangChain, FAISS, sentence-transformers.
- LLM: Groq `llama3-8b-8192`.
- Frontend: Next.js App Router + React + TypeScript.
- Auth: `streamlit-authenticator` (Streamlit) and JWT (FastAPI).

## Repository Map

- `app.py`: Streamlit app (employee chat + admin portal + user management).
- `main.py`: FastAPI API server.
- `retriever.py`: Hybrid retriever and reranking.
- `ingest.py`: Ingestion pipeline and per-tenant index build.
- `auto_ingest.py`: Watchdog-based auto-rebuild watcher.
- `feedback.py`: Feedback log and source reputation weights.
- `drift.py`: Semantic drift detection helpers.
- `eval.py`: Evaluation harness.
- `app/`, `frontend/components/`, `lib/api.ts`: Next.js frontend.
- `run.ps1`: One-command local bootstrap.

## Prerequisites

- Python 3.10+
- Node.js 18+
- A valid `GROQ_API_KEY` in `.env`

Example `.env`:

```env
GROQ_API_KEY=your_key_here
# Optional
# JWT_SECRET_KEY=replace_in_production
# CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
# DAYONE_USE_RERANKER=1
```

## Quick Start

### Recommended (Windows PowerShell)

```powershell
.\run.ps1
```

What `run.ps1` does:

- Creates `.venv` if missing.
- Installs Python and Node dependencies.
- Runs initial ingestion (`ingest.py`).
- Starts watcher, FastAPI, and Next.js.
- Starts Streamlit in the current terminal unless `-NoStreamlit` is used.

Useful flags:

```powershell
.\run.ps1 -SkipInstall
.\run.ps1 -SkipIngest
.\run.ps1 -NoStreamlit
.\run.ps1 -UseSeparateTerminals
```

### Manual Start

```powershell
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe auto_ingest.py
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
npm install
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'
npm run dev
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Default Local Credentials

Derived from current `config.yaml` password hashes:

- Employee: username `john_doe`, password `password123`
- Admin: username `admin_acme`, password `password123`

Production note: rotate cookie key and credentials before external deployment.

## Frontends

### Streamlit (`app.py`)

- Employee chat with confidence labels and source justification.
- Admin portal for tenant file uploads and tenant user management.
- Uses `streamlit-authenticator` with credentials from `config.yaml`.

### Next.js (`app/` + `frontend/components/`)

- Routes: `/`, `/login`, `/chat`, `/admin`.
- JWT stored in `localStorage` (`dayone_token`).
- Chat UI consumes `/api/chat/stream` SSE events.
- Admin dashboard supports uploads, drift summary, and user management.

## API Surface

Auth:

- `POST /auth/login`

Employee/Admin:

- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/feedback`

Admin only:

- `POST /api/admin/upload`
- `GET /api/admin/drift-report`
- `GET /api/admin/users`
- `POST /api/admin/users`
- `PATCH /api/admin/users/{username}`
- `DELETE /api/admin/users/{username}`

Utility:

- `GET /health`

## Multi-Tenant Data Layout

- Source docs: `data/org_<tenant>/...`
- Tenant index: `vector_store/org_<tenant>/index.faiss`
- Auto-ingest watcher monitors `data/org_*/` for `.pdf` and `.csv` changes.

## Evaluation

Baseline:

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme
```

With output file:

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme --output eval_results.json
```

With judge mode:

```powershell
.\.venv\Scripts\python.exe eval.py --org org_acme --judge
```

## Current Test Status (April 18, 2026)

Latest run:

```text
8 passed
```

Notes:

- Test suite currently passes cleanly under pytest; external Streamlit testing warnings are filtered in `pytest.ini`.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Environment Variables

- `GROQ_API_KEY`: required.
- `DAYONE_GROQ_MODEL`: optional, default `llama-3.1-8b-instant`.
- `DAYONE_USE_RERANKER`: `1` (default) or `0`.
- `TENANT_RATE_LIMIT_RPM`: default `30`.
- `TENANT_UPLOAD_LIMIT_PER_DAY`: default `20`.
- `TENANT_MAX_CHUNKS`: default `50000`.
- `ACCESS_TOKEN_EXPIRE_MINUTES`: default `1440`.
- `JWT_SECRET_KEY`: optional, falls back to `config.yaml` cookie key.
- `CORS_ORIGINS`: comma-separated origins.

## Troubleshooting

- Missing GROQ key in Streamlit:
  - Add `GROQ_API_KEY` to `.env` and restart.
- Empty tenant KB errors:
  - Run `ingest.py` and verify documents under `data/org_*`.
- Next.js cannot reach backend:
  - Verify `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`.
- Auto-ingest not triggering:
  - Confirm `auto_ingest.py` is running and files are changed under `data/org_*/`.

## Scope and Safety

- Answers are grounded to retrieved chunks but can still be imperfect.
- Use HR review for policy-critical decisions.
- Repository is suitable for local/dev workflows and prototyping.
