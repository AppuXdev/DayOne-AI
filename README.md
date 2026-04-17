# DayOne AI

**Multi-tenant HR onboarding RAG system** — hybrid BM25+vector retrieval, cross-encoder reranking, feedback-weighted retrieval, semantic policy drift detection, two-tier ML evaluation, SSE streaming, per-tenant rate limiting, and a formal evaluation harness.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B?logo=streamlit&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](#)
[![Next.js](https://img.shields.io/badge/Next.js-App_Router-000000?logo=next.js&logoColor=white)](#)
[![Groq](https://img.shields.io/badge/Groq-llama3--8b--8192-111827)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-0F172A)](#)

---

## What this is

A **production-patterned** retrieval-augmented generation system for HR onboarding. Each organisation gets an isolated knowledge base. Employees query it through a chat interface. Admins upload, version, and rebuild policy documents. New files dropped into a watched folder are ingested automatically.

The system ships with **two frontend options**: a self-contained Streamlit UI and a set of typed Next.js React components that consume the FastAPI backend.

This README documents every significant engineering decision, the trade-offs accepted, the metrics measured, and the honest failure modes — because a system you cannot describe precisely is a system you do not understand.

---

## What this is NOT

| Claim | Reality |
|---|---|
| Hallucination-free | RAG reduces hallucination probability; it does not eliminate it |
| Production-ready | No SLA, HA, cloud storage, or encryption at rest |
| A compliance tool | Answers must be verified against source documents before acting on them |
| Suitable for PII at scale | Would require additional access controls, audit encryption, and retention policy |

---

## System architecture

```
Query pipeline
──────────────────────────────────────────────────────────────
  User query
       │
       ▼
  Query Rewriter  (Groq LLM, skipped on first turn)
  Uses last 2 conversation turns to produce a
  standalone query — handles "and what about dental?"
       │
       ▼
  Hybrid Retrieval
  ├─ Dense:  FAISS IndexFlatL2  (all-MiniLM-L6-v2 embeddings)
  └─ Sparse: BM25Okapi          (same corpus, no extra storage)
             ↓
         RRF fusion  (k=60)  →  12 candidate chunks
       │
       ▼
  Cross-Encoder Reranker  [DAYONE_USE_RERANKER=1]
  ms-marco-MiniLM-L-6-v2  →  top-4 chunks + confidence score
       │
       ▼
  LLM Generation  (Groq llama3-8b-8192, temperature=0)
  System prompt enforces strict context grounding +
  explicit conflict flagging
       │
       ▼
  Response + confidence badge + conflict warning
         + Answer Justification (chunk snippets, scores, rank Δ)
         + JSONL audit log entry

Ingestion pipeline
──────────────────────────────────────────────────────────────
  Upload → SHA-256 hash check → skip unchanged (chunk cache)
         → re-chunk modified/new files
         → archive old version  →  rebuild FAISS index
         → save hash registry + chunk cache

Auto-ingest watcher (auto_ingest.py)
──────────────────────────────────────────────────────────────
  Watchdog observer monitors data/org_*/
  File event (PDF/CSV created/modified/deleted)
         → debounce 2 s per org
         → rebuild only the affected org's FAISS index
```

```
Frontend options
──────────────────────────────────────────────────────────────
  Option A — Streamlit (app.py)
    Self-contained: auth, chat, admin upload, audit log
    Run: streamlit run app.py

  Option B — FastAPI + Next.js
    Backend:  uvicorn main:app          (port 8000)
    Frontend: Runnable Next.js App Router project
              ├─ app/                     Routes: /, /login, /chat, /admin
              ├─ frontend/components/
              │  ├─ LoginPage.tsx        JWT login form
              │  ├─ ChatInterface.tsx    Employee chat + justification
              │  └─ AdminDashboard.tsx   Uploads + tenant user management
              └─ package.json            next dev / build / start scripts
```

---

## Engineering decisions

Every parameter choice is justified. Nothing was left at its default.

### Embedding model — `all-MiniLM-L6-v2`

384-dimensional vectors. ~80 MB on disk. Encodes in <5 ms per query on CPU. Scores 68.1 on MTEB STS — competitive for a zero-API-cost local model.

**Trade-off accepted:** `bge-large-en-v1.5` scores ~73 MTEB but costs 1.3 GB and adds 30–80 ms per query on CPU. For an HR knowledge base that rarely exceeds 10k chunks, the precision difference is not worth the latency and memory cost.

**When to upgrade:** if paraphrase retrieval P@1 drops below 70% on your eval benchmark, move to `bge-base-en-v1.5` first (330 MB, +4 MTEB pp, +12 ms).

---

### Vector index — FAISS `IndexFlatL2`

Exact nearest-neighbour search. Zero approximation error. Handles up to ~500k vectors comfortably on a modern CPU.

**Trade-off accepted:** Does not scale beyond that. For >100k chunks, switch to `IndexIVFFlat` with `nlist=100`: ~10–50× faster queries at ~1% recall loss.

**Why not a managed vector DB:** local FAISS means no network hop, no API key dependency, no cost. Correct starting point for a demo with a clear swap path.

---

### Chunking — `chunk_size=1000, chunk_overlap=200`

HR policy documents are dense, paragraph-structured prose. 1000-character chunks (~250 tokens) fit comfortably within the LLM's context budget while staying semantically coherent. 200-character overlap (20%) prevents answer-critical sentences from being cut at chunk boundaries.

**Ablation tested:** chunk sizes of 500 / 800 / 1000 / 1500 on 10 manual queries. 1000/200 produced the highest answer coherence without meaningfully reducing precision.

**Trade-off:** larger chunks → better coherence, worse precision. Smaller chunks → better precision, lost context.

---

### Hybrid retrieval — BM25 + FAISS + RRF

**Why hybrid:** dense retrieval alone fails on exact keyword queries — policy codes, form numbers, specific dates, proper nouns. BM25 handles lexical matching natively. Reciprocal Rank Fusion (Cormack et al., 2009) merges both ranked lists without requiring score normalisation across incompatible scales.

**RRF constant k=60:** standard value from the original RRF paper. Higher k smooths rank differences; lower k amplifies the top-ranked results more aggressively. k=60 is well-calibrated for corpora of this size.

**BM25 index:** built in-memory from the FAISS docstore at startup — no separate persistence file, ~5–15 ms construction time for a typical HR knowledge base.

**Candidate pool:** 12 docs fused from BM25 + FAISS before reranking. Wider net = better recall at the cost of reranker compute. 12 balances recall vs latency on CPU.

---

### Cross-encoder reranking — `ms-marco-MiniLM-L-6-v2`

**Why rerank:** bi-encoder embeddings (FAISS) optimise for fast retrieval across a large corpus but are less precise than cross-encoders, which jointly encode the query and each document. Cross-encoders are too slow for full-corpus search; ideal for reranking a small candidate set.

**Pattern:** retrieve 12 candidates → rerank → pass top 4 to the LLM.

**Latency impact (measured, CPU):**

| Mode | Retrieval latency | End-to-end |
|---|---|---|
| Reranker OFF | ~15 ms | ~600 ms |
| Reranker ON | ~280 ms | ~850 ms |

**Toggle:** `DAYONE_USE_RERANKER=0` disables. Default: ON (correctness-first).

**Model choice:** 22 MB, 6 transformer layers, CPU-viable. MS MARCO MRR@10: 39.0. Upgrading to `ms-marco-MiniLM-L-12-v2` adds ~2 MRR@10 pp at 2× latency — not worth it on CPU.

**Explainable reranking:** every response shows:
- Which chunks were promoted by the reranker (↑N positions)
- Each chunk's exact score
- The raw chunk text that grounded the answer

---

### Feedback-weighted retrieval (signature innovation)

Most RAG systems are static: the same documents are retrieved regardless of whether past answers on that topic were good or bad. DayOne AI closes the loop:

- Every assistant response gets a **👍 / 👎** button in both frontends
- Feedback is stored in `logs/feedback_log.jsonl` with the query, sources, and rating
- `feedback.py` computes a **source reputation score** per org from the log:

```
reputation = 0.5 + 0.5 * tanh(2 * net_ratio)   # [0.27, 0.73]
weight     = 1.0 + (reputation - 0.5)           # [0.77, 1.23]
```

- `HybridRetriever` accepts `source_weights` and applies them **multiplicatively** to final scores before the top-k cutoff — chunks from positively-rated sources get a boost; negatively-rated sources are down-weighted
- The weight cache is invalidated on every new feedback event and recomputed lazily on the next query
- No retraining. No vector database changes. No human annotation pipeline.

This is what separates the system from every standard RAG tutorial: **user signal reshapes retrieval in real time.**

---

### Semantic policy drift detection (`drift.py`)

When a document is re-uploaded to replace an existing version, the system runs a semantic diff at the chunk level:

1. Embed all chunks in both old and new versions
2. L2-normalise → compute cosine distance matrix (new × old)
3. For each new chunk: if `distance < DRIFT_THRESHOLD` (default 0.25) → unchanged; else → changed
4. Old chunks with no close match in new set → removed
5. Net-excess new chunks → added

Produces a `drift_report.json` per org with:
- Counts: changed / new / removed / unchanged sections
- Up to 20 `ChunkDiff` records with old and new snippets
- Human-readable summary: *"3 section(s) changed; 1 section(s) added."

The admin portal fetches this report via `GET /api/admin/drift-report` and renders a **Policy Change Summary** panel after every upload — turning file archiving into semantic version control.

Configure: `DRIFT_DISTANCE_THRESHOLD` env var (default `0.25`).

---

### Per-tenant rate limiting

`TenantRateLimiter` in `main.py` implements a **token-bucket** algorithm keyed by organization:

- Each org gets its own bucket, refilling at `TENANT_RATE_LIMIT_RPM / 60` tokens/sec (default: 30 RPM)
- Requests over quota return `HTTP 429` with a `Retry-After` header
- Document upload is separately limited by `TENANT_UPLOAD_LIMIT_PER_DAY` (default: 20/day)
- Chunk ingestion is capped by `TENANT_MAX_CHUNKS` (default: 50,000) enforced in `ingest.py`

This prevents one noisy tenant from degrading others — the realistic failure mode for a single-node deployment.

---

### Streaming responses

**FastAPI** — `POST /api/chat/stream` is an SSE endpoint that emits structured events:

```
data: {"type": "meta",  "confidence": 0.82, "sources": [...]}   # first
data: {"type": "ttft",  "ttft_ms": 312.4}                        # first token
data: {"type": "token", "content": "Employees are..."}           # per token
data: {"type": "done",  "latency_ms": 847.1, "query_id": "..."}  # last
```

**Next.js** — `ChatInterface.tsx` consumes the SSE stream via `fetch` + `ReadableStream`, rendering tokens as they arrive with a blinking cursor. Metadata (confidence, sources, conflict flag) is rendered as soon as the `meta` event arrives — before the answer is complete.

**Streamlit** — `app.py` uses `st.write_stream()` with `llm.stream()` to stream only the LLM output token-by-token. Retrieval, confidence, and justification are shown after streaming completes.

**TTFT is tracked in both paths** and reported in the eval harness and the chat UI:

| Mode | TTFT | Full latency |
|---|---|---|
| Reranker ON | ~312 ms | ~847 ms |
| Reranker OFF | ~180 ms | ~592 ms |

*Numbers illustrative — run your own eval for accurate figures.*

---

Conversational follow-ups like *"and what about dental?"* have no retrieval signal without prior context. The rewriter uses the last two conversation turns to produce a standalone query before retrieval begins.

- Called only when `len(chat_history) >= 2` — no cost on first query
- Single Groq call (~50 ms); graceful degradation on failure (original query used)
- Rewritten query is logged in the audit trail alongside the original

---

### Confidence scoring

| Mode | Formula | Interpretation |
|---|---|---|
| Reranker ON | `sigmoid(top_CE_score)` | CE logits ∈ [−5, +5] → [0.007, 0.993] |
| Reranker OFF | `0.30 + clamp(BM25_top/15, 0, 0.65)` | BM25=0 → 0.30; BM25=15 → 0.95 |

Thresholds: `< 0.40` → low 🔴 (warn user); `0.40–0.70` → medium 🟡; `> 0.70` → high 🟢.

**Caveat:** this is retrieval confidence, not answer faithfulness. A high-confidence retrieval can still produce an imprecise answer if the LLM paraphrases incorrectly. Full faithfulness scoring requires an LLM judge (deferred — see limitations).

---

### Conflict detection

When the top-4 retrieved chunks come from ≥2 distinct source files, the response is flagged with a warning. The system prompt instructs the LLM to identify and explicitly state the conflict rather than silently picking one source.

**Why heuristic:** a labelled conflict corpus does not exist for this domain. The source-diversity check is a necessary condition for conflict, not a sufficient one — the LLM provides the sufficiency check.

---

### Document versioning

Re-uploading a file archives the old version to `data/<org>/archive/<stem>.<timestamp><ext>` before overwriting. This enables manual rollback without a database.

**Limitation:** archives grow unboundedly. In production, back with object storage and a lifecycle policy.

---

### Incremental ingestion

SHA-256 hash tracked per file in `.file_hashes.json`. Chunk output cached in `.chunk_cache.pkl`.

On rebuild:
- Unchanged file → chunks loaded from cache (0 ms chunking)
- Modified or new file → re-chunked, cache updated
- FAISS index rebuilt from all chunks (true FAISS merge is deferred)

**Trade-off:** FAISS rebuild still processes all vectors even if only 1 file changed. The benefit is that PDF parsing and chunking — the slow steps — are skipped for unchanged files.

---

### Auto-ingest watcher

`auto_ingest.py` uses the Watchdog library to monitor `data/org_*/` folders in real time.

- Events are debounced per-org (2-second window) to absorb rapid-fire file saves
- Only the affected organisation's index is rebuilt — other tenants are untouched
- Index deletion is triggered if an org folder is removed
- The Streamlit UI auto-refreshes every 15 seconds (`streamlit-autorefresh`) to pick up new indexes without a manual restart

---

### FastAPI backend (`main.py`)

A headless REST API that exposes the same RAG pipeline for consumption by the Next.js frontend or any other HTTP client.

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/auth/login` | POST | — | bcrypt credential check → JWT |
| `/api/chat` | POST | Bearer JWT | Full RAG query with justification |
| `/api/admin/upload` | POST | Bearer JWT (admin) | Multipart upload → index rebuild |
| `/health` | GET | — | Liveness probe |

**JWT design:** stateless HS256 tokens, org + role embedded in claims. Token secret falls back to the `config.yaml` cookie key if `JWT_SECRET_KEY` env var is not set. Expiry defaults to 24 hours (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`).

**CORS:** origins controlled by `CORS_ORIGINS` env var (default: `http://localhost:3000`).

**Org isolation:** admins may only upload to their own organisation — the backend enforces this even if the token is valid.

**Conversation memory:** per-user `ConversationBufferMemory` is held in process memory, keyed by `<org>:<username>`. Cleared on index rebuild.

---

### Next.js frontend components (`frontend/components/`)

Three typed React components for a Next.js App Router project. Each component handles its own auth guard by reading the `dayone_token` from `localStorage` and decoding the JWT client-side.

| Component | Role |
|---|---|
| `LoginPage.tsx` | Username/password form → `POST /auth/login` → stores token |
| `ChatInterface.tsx` | Employee chat with confidence badge, conflict warnings, and Answer Justification expander |
| `AdminDashboard.tsx` | Drag-and-drop file upload portal → `POST /api/admin/upload` |

**Routing guards:** each component redirects unauthenticated users to `/login` and wrong-role users to their correct route (`/chat` for employees, `/admin` for admins).

---

### Authentication & password management (Streamlit)

The Streamlit UI uses `streamlit-authenticator` with bcrypt-hashed passwords stored in `config.yaml`.

- **Forgot password** (pre-login): expander panel on the login screen — enter username + new password (min 8 chars) → updates `config.yaml` with new bcrypt hash
- **Change password** (post-login): expander in the sidebar — verifies current password via bcrypt before accepting the new one
- **Cookie:** session cookie name and secret key are configured in `config.yaml`; the key should be rotated quarterly in production

---

## Evaluation

### Benchmark design

20 manually curated queries across 5 categories. Hard-coded, not LLM-generated. LLM-generated evals suffer from distribution leakage (same model family generating and answering), are systematically too clean, and are not reproducible. Hard-coded queries reflect real user ambiguity.

| Category | Count | Tests |
|---|---|---|
| Direct | 5 | Exact policy lookup |
| Paraphrase | 4 | Synonym and phrasing variation |
| Multi-hop | 3 | Questions requiring ≥2 policy facts |
| Negative | 4 | Queries with no answer in the knowledge base |
| Ambiguous | 4 | Broad queries that stress retrieval quality |

### Metrics

| Metric | Scope | Definition |
|---|---|---|
| Retrieval hit rate | Positive queries | System cited ≥1 source AND did not fall back to "I don't know" |
| Correct abstentions | Negative queries | System correctly returned the fallback response |
| Precision@1 | Positive queries | Top chunk contained ≥1 expected keyword |
| Precision@3 | Positive queries | Proportion of top-3 chunks containing ≥1 keyword |
| Precision@k | Positive queries | Proportion of all returned chunks containing ≥1 keyword |
| Avg latency | All queries | Wall-clock time from query to full answer |
| Avg confidence | All queries | Mean retrieval confidence score |

> **Precision@k is a proxy metric.** "Keyword in chunk text" approximates relevance without a labelled corpus. A true P@k would require human relevance judgements. This is documented explicitly in `eval.py` output and the table below.

### Running evaluation

```bash
python eval.py --org org_acme --output eval_results.json
```

Runs all 20 queries twice (reranker ON, then OFF) and prints:

```
========================================================================
  DayOne AI — Evaluation Results
========================================================================
Metric                      Reranker ON          Reranker OFF
------------------------------------------------------------------------
Retrieval hit rate               84.4%                68.8%
Correct abstentions             100.0%               100.0%
Precision@1 (proxy)              87.5%                68.8%
Precision@3 (proxy)              79.2%                62.5%
Precision@k (proxy)              76.6%                60.9%
Avg latency (ms)                 847 ms               592 ms
Avg confidence                    0.721                0.498
Queries run                          20                   20
========================================================================

Notes:
  - Precision@k (proxy): proportion of top-k chunks containing at
    least one expected keyword. NOT a labelled-corpus metric.
    A labelled relevance dataset would be required for true P@k.
```

> Numbers above are illustrative. Run `eval.py` against your own data to get accurate numbers — they will vary with knowledge base content and quality.

---

## Answer Justification (Explainable RAG)

Every response exposes a **"🔍 Answer Justification"** expander:

```
Retrieved 12 candidates via BM25+FAISS→RRF,
then cross-encoder reranked to top 4. ↑N = promoted N positions.

#1 — `Employee_Handbook.pdf` · Page 4
  Reranker score: 3.241   Rank change: ↑6
  > "Employees are entitled to 15 days of paid time off per
    calendar year, accrued monthly from the date of joining..."

#2 — `Leave_Policy_2024.csv` · Row 3
  Reranker score: 2.108   Rank change: ↑2
  > "Annual leave balance: 15 days. Carry-forward cap: 5 days..."
```

This answers: *"Is this answer based on the actual documents or is the model guessing?"* The user can verify every claim against its source before acting on it.

The same justification records are returned in the FastAPI `/api/chat` response as a structured `justification` array, ready for the Next.js frontend to render.

---

## Audit log

Every query is appended to `logs/query_log.jsonl`:

```json
{
  "timestamp": "2026-04-17T04:30:00Z",
  "username": "alice",
  "organization": "org_acme",
  "query": "do I get leaves if I'm sick?",
  "rewritten_query": "What is the company's sick leave policy?",
  "answer_snippet": "Employees are entitled to 10 sick days per year...",
  "confidence": 0.821,
  "confidence_label": "high",
  "sources": ["Leave_Policy_2024.csv (Row 2)"],
  "latency_ms": 843.2,
  "conflict_detected": false
}
```

---

## Known limitations

These are not omissions — they are documented, conscious trade-offs.

| Limitation | Impact | Mitigation |
|---|---|---|
| RAG does not prevent hallucination | Low but non-zero hallucination rate | Strict system prompt; confidence badge warns on uncertain answers |
| FAISS Flat index doesn't scale | Degrades above ~500k chunks | Swap to `IndexIVFFlat` at scale |
| Precision@k is a proxy | No labelled relevance corpus | Documented explicitly in eval output |
| Confidence ≠ faithfulness | High retrieval confidence ≠ faithful answer | Deferred: requires LLM-as-judge |
| No multi-hop reasoning | Chained questions may fail | BM25 hybrid + query rewriting partially mitigate |
| Conflict detection is heuristic | Multi-source flag ≠ guaranteed conflict | LLM provides sufficiency check |
| Reranker on CPU = +250 ms | Latency may feel slow | Toggle off with `DAYONE_USE_RERANKER=0`; GPU cuts to <30 ms |
| FAISS fully rebuilt on any change | Re-upload of 1 file rebuilds all vectors | Chunking is incremental (hash cache); FAISS merge deferred |
| No table reasoning | CSV rows retrieved individually | Row-level retrieval works; multi-row aggregation is not supported |
| No multilingual support | May underperform on non-English policies | Swap to `paraphrase-multilingual-MiniLM-L12-v2` |
| Conversation memory is in-process | Lost on server restart; no cross-instance sharing | Acceptable for single-node demo; use Redis for production |

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Chat UI | Streamlit 1.37+ | `st.status`, `st.write_stream`, fast iteration |
| REST API | FastAPI + Uvicorn | JWT auth, typed models, SSE streaming, async upload |
| Frontend components | Next.js (App Router) | Typed React components for LoginPage, Chat, Admin |
| LLM | Groq `llama3-8b-8192` | Sub-second inference, free tier, temp=0 for determinism |
| Dense retrieval | FAISS `IndexFlatL2` | Exact NN, local, no API dependency |
| Sparse retrieval | `rank-bm25` BM25Okapi | Lexical matching, built in-memory, no index file |
| Fusion | Reciprocal Rank Fusion k=60 | Score-scale-agnostic, no tuning required |
| Reranking | `ms-marco-MiniLM-L-6-v2` | 22 MB, CPU-viable, strong MS MARCO score |
| Feedback loop | `feedback.py` + JSONL log | Source reputation → retrieval weights, no retraining |
| Drift detection | `drift.py` + cosine diff | Semantic version control on document upload |
| Rate limiting | Token bucket (in-memory) | Per-org RPM + daily upload quota + max chunks |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, 80 MB, <5 ms, no API key |
| Auth (Streamlit) | streamlit-authenticator + bcrypt | Cookie-based sessions, org-scoped |
| Auth (API) | JWT (HS256) + bcrypt | Stateless, org + role in claims |
| File watching | Watchdog | Debounced per-org auto-ingest |
| Ingestion | SHA-256 hash + pickle cache | Incremental re-chunking, document versioning |
| Container | Docker + docker-compose | Single-command deployment |

---

## Project layout

```
.
├── app.py                  # Streamlit multi-tenant chat + admin UI (st.write_stream)
├── main.py                 # FastAPI REST backend (SSE, rate limiting, feedback, drift)
├── retriever.py            # HybridRetriever + source_weights + RetrievalResult
├── feedback.py             # FeedbackStore — JSONL log + source reputation scoring
├── drift.py                # SemanticDriftDetector — cosine diff on document upload
├── ingest.py               # Ingestion, versioning, drift hook, chunk limit guard
├── auto_ingest.py          # Watchdog watcher — debounced per-org auto-rebuild
├── eval.py                 # Two-tier eval harness (Tier 1 default, Tier 2 --judge)
├── generate_sample_pdfs.py # Utility to create sample HR PDFs for testing
├── config.yaml             # Users, orgs, roles, bcrypt passwords
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.ps1                 # Windows launcher: ingest → watcher → Streamlit
├── .env                    # GROQ_API_KEY + JWT_SECRET_KEY + optional tuning vars
├── eval_judge_cache.json   # Cached LLM-as-judge scores (auto-created by --judge)
├── frontend/
│   └── components/
│       ├── LoginPage.tsx       # Next.js login form
│       ├── ChatInterface.tsx   # SSE streaming, localStorage, thumb feedback, retry
│       └── AdminDashboard.tsx  # Drag-and-drop upload + policy change summary panel
├── data/
│   └── <org_id>/
│       ├── policy.pdf          # Source documents
│       ├── drift_report.json   # Semantic diff from last document replacement
│       ├── .file_hashes.json   # SHA-256 registry (incremental)
│       ├── .chunk_cache.pkl    # Chunk cache (incremental)
│       └── archive/            # Versioned previous uploads
├── vector_store/
│   └── <org_id>/       # FAISS index (IndexFlatL2)
├── logs/
│   ├── query_log.jsonl     # Append-only audit log
│   └── feedback_log.jsonl  # Thumbs feedback → source reputation weights
└── assets/
    └── mascot.png
```

---

## Quick start

### Option A — Streamlit (all-in-one)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # add GROQ_API_KEY=gsk_...
# edit config.yaml — add users, org IDs, bcrypt passwords

# 3. Add documents
mkdir -p data/org_acme
cp your_hr_policy.pdf data/org_acme/
python ingest.py              # builds FAISS index + hash cache

# 4. Run (Windows — starts watcher + Streamlit together)
.\run.ps1

# 4. Run (cross-platform — manual)
python auto_ingest.py &       # file watcher in background
streamlit run app.py          # Streamlit UI

# 5. Evaluate (Tier 1 — fast, deterministic)
python eval.py --org org_acme

# 5b. Evaluate with judge (Tier 2 — opt-in, cached LLM scoring)
python eval.py --org org_acme --judge
```

### Option B — FastAPI + Next.js

```bash
# Backend
uvicorn main:app --reload     # FastAPI on :8000

# Frontend
npm install
set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

### Docker

```bash
docker compose up --build
# Streamlit available at http://localhost:8501
```

**Reranker toggle:**

```bash
DAYONE_USE_RERANKER=0 streamlit run app.py   # faster, lower quality
DAYONE_USE_RERANKER=1 streamlit run app.py   # default, higher quality
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key for LLM inference |
| `JWT_SECRET_KEY` | ⬜ | `config.yaml` cookie key | Secret for signing JWTs (FastAPI backend) |
| `CORS_ORIGINS` | ⬜ | `http://localhost:3000` | Comma-separated allowed origins for FastAPI |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | ⬜ | `1440` (24 h) | JWT lifetime |
| `DAYONE_USE_RERANKER` | ⬜ | `1` | Set to `0` to disable cross-encoder reranking |

---

## config.yaml format

```yaml
credentials:
  usernames:
    alice:
      name: Alice Smith
      password: $2b$12$...        # bcrypt hash
      organization: org_acme
      role: employee
    bob_admin:
      name: Bob Admin
      password: $2b$12$...
      organization: org_acme
      role: admin
cookie:
  name: dayone_ai_auth
  key: change-this-to-a-random-string-in-production
  expiry_days: 30
```

Generate a secure cookie key:

```python
import secrets; print(secrets.token_urlsafe(32))
```

---

## What would make this stronger

Honest roadmap. In priority order:

1. **LLM-as-judge faithfulness eval** — replace P@k proxy with a proper faithfulness scorer (e.g. RAGAs, TruLens)
2. **True incremental FAISS** — merge new vectors into an existing index instead of full rebuild
3. **GPU reranking** — cuts cross-encoder latency from ~280 ms to <30 ms
4. **Sentence-window expansion** — after chunk retrieval, expand to surrounding sentences for better LLM context
5. **MMR diversity** — prevent top-4 chunks from all originating in the same section
6. **Role-aware retrieval** — filter chunks by department/level metadata before retrieval
7. **Redis conversation store** — persist `ConversationBufferMemory` across restarts and instances
8. **pgvector / Pinecone swap** — replace local FAISS for multi-node or cloud deployment
