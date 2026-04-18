"""DayOne AI FastAPI backend.

Headless API for multi-tenant HR RAG with JWT auth, LangChain + Groq chat,
and admin uploads that rebuild per-organization FAISS indexes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from bcrypt import checkpw
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ingest import archive_file_if_exists, rebuild_organization_index
from retriever import HybridRetriever, RetrievalResult, USE_RERANKER, CONF_LOW, confidence_label
from feedback import get_feedback_store
from drift import load_drift_report, DriftReport
from user_admin import (
    ROLE_ADMIN,
    ROLE_EMPLOYEE,
    clone_config,
    create_user_record,
    delete_user_record,
    load_app_config,
    save_app_config,
    serialize_user,
    update_user_record,
)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
LOGS_DIR = ROOT_DIR / "logs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3-8b-8192"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
TENANT_RATE_LIMIT_RPM: int = int(os.getenv("TENANT_RATE_LIMIT_RPM", "30"))
TENANT_UPLOAD_LIMIT_PER_DAY: int = int(os.getenv("TENANT_UPLOAD_LIMIT_PER_DAY", "20"))

SYSTEM_PROMPT = (
    "You are DayOne AI, a professional HR onboarding assistant. "
    "Answer ONLY from the retrieved context provided. "
    "If the answer is not in the context, say exactly: "
    "'I do not have that information in the current HR files. Please contact HR.' "
    "Do not invent, infer, or extrapolate beyond the retrieved text. "
    "If the retrieved context contains conflicting information from different "
    "documents, explicitly flag the conflict before answering."
)

app = FastAPI(title="DayOne AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)
conversation_memories: Dict[str, ConversationBufferMemory] = {}


# ---------------------------------------------------------------------------
# Per-tenant rate limiter (token bucket)
# ---------------------------------------------------------------------------

class TenantRateLimiter:
    """In-process token-bucket rate limiter keyed by organization.

    Each org gets its own bucket that refills at `rate_rpm / 60` tokens/sec.
    When the bucket is empty, the request receives HTTP 429 with Retry-After.
    This prevents one noisy tenant from exhausting server resources on a
    single-node deployment — the realistic failure mode for this architecture.
    """

    def __init__(self, rate_rpm: int = TENANT_RATE_LIMIT_RPM) -> None:
        self._rate = rate_rpm / 60.0   # tokens per second
        self._capacity = float(rate_rpm)
        self._buckets: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, org_id: str) -> None:
        """Raise HTTP 429 if the org's token bucket is empty."""
        now = time.monotonic()
        with self._lock:
            last = self._timestamps.get(org_id, now)
            tokens = self._buckets.get(org_id, self._capacity)
            elapsed = now - last
            tokens = min(self._capacity, tokens + elapsed * self._rate)
            self._timestamps[org_id] = now
            if tokens < 1.0:
                retry_after = int((1.0 - tokens) / self._rate) + 1
                self._buckets[org_id] = tokens
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for organisation '{org_id}'. "
                           f"Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            self._buckets[org_id] = tokens - 1.0


_rate_limiter = TenantRateLimiter()


# Per-org upload counters (day-scoped, in-process)
_upload_counts: Dict[str, Dict[str, int]] = {}  # org -> {date_str: count}
_upload_lock = threading.Lock()


def _check_upload_limit(org_id: str) -> None:
    """Raise HTTP 429 if the org has exceeded its daily upload quota."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _upload_lock:
        day_counts = _upload_counts.setdefault(org_id, {})
        count = day_counts.get(today, 0)
        if count >= TENANT_UPLOAD_LIMIT_PER_DAY:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily upload limit ({TENANT_UPLOAD_LIMIT_PER_DAY}) reached for '{org_id}'.",
            )
        day_counts[today] = count + 1


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    organization: str
    role: str
    expires_at: datetime


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    token: Optional[str] = None


class SourceMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    row: Optional[int] = None
    tenant: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    username: str
    organization: str
    role: str
    model: str
    confidence: float = 0.0
    confidence_label: str = "low"
    conflict_detected: bool = False
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    justification: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "ok"
    query_id: str = ""   # used by frontend to submit feedback


class UploadResponse(BaseModel):
    organization: str
    saved_files: List[str]
    rebuilt: bool
    message: str
    drift_summary: Optional[str] = None


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1)
    rating: str = Field(pattern=r"^(up|down)$")
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    query: str = ""


class TokenPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub: str
    username: str
    organization: str
    role: str
    exp: Optional[int] = None


class InMemoryUser(BaseModel):
    username: str
    name: str = ""
    email: str = ""
    organization: str
    role: str = "employee"
    password: str


class ManagedUser(BaseModel):
    username: str
    name: str = ""
    email: str = ""
    organization: str
    role: str = ROLE_EMPLOYEE


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)
    name: str = ""
    email: str = ""
    role: str = ROLE_EMPLOYEE


class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=8)


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    return load_app_config(CONFIG_PATH)


@lru_cache(maxsize=1)
def get_jwt_secret() -> str:
    config = load_config()
    cookie = config.get("cookie", {}) if isinstance(config, dict) else {}
    fallback = str(cookie.get("key", "change-this-secret-key"))
    return JWT_SECRET_KEY or fallback


@lru_cache(maxsize=1)
def get_users() -> Dict[str, Dict[str, Any]]:
    config = load_config()
    credentials = config.get("credentials", {}) if isinstance(config, dict) else {}
    usernames = credentials.get("usernames", {}) if isinstance(credentials, dict) else {}
    return usernames if isinstance(usernames, dict) else {}


def persist_config(config: Dict[str, Any]) -> None:
    save_app_config(CONFIG_PATH, config)
    load_config.cache_clear()
    get_jwt_secret.cache_clear()
    get_users.cache_clear()


def require_admin_user(current_user: TokenPayload) -> None:
    if current_user.role != ROLE_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")


@lru_cache(maxsize=1)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def sanitize_filename(filename: str) -> str:
    return Path(filename).name.replace(" ", "_")


def create_access_token(*, username: str, organization: str, role: str) -> LoginResponse:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "username": username,
        "organization": organization,
        "role": role,
        "exp": expire,
    }
    token = jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)
    return LoginResponse(
        access_token=token,
        username=username,
        organization=organization,
        role=role,
        expires_at=expire,
    )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def authenticate_user(username: str, password: str) -> InMemoryUser:
    users = get_users()
    record = users.get(username)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    stored_password = str(record.get("password", ""))
    if not verify_password(password, stored_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    organization = str(record.get("organization", "")).strip()
    role = str(record.get("role", "employee") or "employee").strip().lower()
    if not organization:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User has no organization assigned")

    return InMemoryUser(
        username=username,
        name=str(record.get("name", "")),
        email=str(record.get("email", "")),
        organization=organization,
        role=role,
        password=stored_password,
    )


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> TokenPayload:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return decode_token(credentials.credentials)


@lru_cache(maxsize=64)
def load_vector_store_for_org(org_id: str, org_signature: str) -> Optional[FAISS]:
    if org_signature in {"missing", "empty"}:
        return None

    org_store = VECTOR_STORE_DIR / org_id
    if not org_store.exists():
        return None

    try:
        return FAISS.load_local(
            str(org_store),
            load_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def compute_org_signature(org_id: str) -> str:
    org_dir = DATA_DIR / org_id
    if not org_dir.exists():
        return "missing"

    signatures: List[str] = []
    for file_path in sorted(org_dir.rglob("*.pdf")) + sorted(org_dir.rglob("*.csv")):
        stat = file_path.stat()
        signatures.append(f"{file_path.relative_to(org_dir)}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(signatures) if signatures else "empty"


def get_memory_key(username: str, organization: str) -> str:
    return f"{organization}:{username}"


def get_or_create_memory(username: str, organization: str) -> ConversationBufferMemory:
    key = get_memory_key(username, organization)
    memory = conversation_memories.get(key)
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        conversation_memories[key] = memory
    return memory


def build_hybrid_retriever(
    vector_store: FAISS,
    embeddings: HuggingFaceEmbeddings,
    source_weights: Optional[Dict[str, float]] = None,
) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        use_reranker=USE_RERANKER,
        source_weights=source_weights,
    )


def rewrite_query(query: str, memory: ConversationBufferMemory, llm: ChatGroq) -> str:
    """Contextualise a follow-up query using recent chat history."""
    messages = memory.chat_memory.messages
    if len(messages) < 2:
        return query
    recent = messages[-4:]
    history = "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {m.content}"
        for i, m in enumerate(recent)
    )
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Follow-up question: {query}\n\n"
        "Rewrite as a fully self-contained question. Output ONLY the rewritten question."
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def detect_conflict(docs: List[Any]) -> bool:
    sources = {Path(str(d.metadata.get("source", ""))).name for d in docs}
    return len(sources) >= 2


def write_audit_log(
    username: str, org_id: str, query: str, rewritten_query: str,
    answer: str, confidence: float, sources: List[str],
    latency_ms: float, conflict_detected: bool,
) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "organization": org_id,
        "query": query,
        "rewritten_query": rewritten_query,
        "answer_snippet": answer[:200],
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "sources": sources,
        "latency_ms": round(latency_ms, 1),
        "conflict_detected": conflict_detected,
    }
    log_path = LOGS_DIR / "query_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def serialize_source_document(document: Any) -> SourceMetadata:
    metadata = getattr(document, "metadata", {}) or {}
    source_path = Path(str(metadata.get("source", "unknown")))
    extra_metadata = {key: value for key, value in metadata.items() if key not in {"source", "page", "row", "tenant"}}
    return SourceMetadata(
        source=source_path.name,
        page=int(metadata["page"]) if str(metadata.get("page", "")).isdigit() else metadata.get("page"),
        row=int(metadata["row"]) if str(metadata.get("row", "")).isdigit() else metadata.get("row"),
        tenant=str(metadata.get("tenant", "")) or None,
        metadata={"source_path": str(source_path), **extra_metadata},
    )


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    user = authenticate_user(payload.username.strip(), payload.password)
    return create_access_token(username=user.username, organization=user.organization, role=user.role)


@app.get("/api/admin/users", response_model=List[ManagedUser])
def list_admin_users(current_user: TokenPayload = Depends(get_current_user)) -> List[ManagedUser]:
    require_admin_user(current_user)
    org_users: List[ManagedUser] = []
    for username, record in get_users().items():
        if str(record.get("organization", "")).strip() != current_user.organization:
            continue
        org_users.append(ManagedUser(**serialize_user(username, record)))
    return sorted(org_users, key=lambda user: user.username.lower())


@app.post("/api/admin/users", response_model=ManagedUser, status_code=status.HTTP_201_CREATED)
def create_admin_user(
    payload: UserCreateRequest,
    current_user: TokenPayload = Depends(get_current_user),
) -> ManagedUser:
    require_admin_user(current_user)
    config = clone_config(load_config())
    try:
        created = create_user_record(
            config=config,
            username=payload.username,
            password=payload.password,
            organization=current_user.organization,
            role=payload.role,
            name=payload.name,
            email=payload.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    persist_config(config)
    return ManagedUser(**created)


@app.patch("/api/admin/users/{username}", response_model=ManagedUser)
def update_admin_user(
    username: str,
    payload: UserUpdateRequest,
    current_user: TokenPayload = Depends(get_current_user),
) -> ManagedUser:
    require_admin_user(current_user)
    config = clone_config(load_config())
    try:
        updated = update_user_record(
            config=config,
            username=username,
            current_organization=current_user.organization,
            name=payload.name,
            email=payload.email,
            role=payload.role,
            password=payload.password,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    persist_config(config)
    return ManagedUser(**updated)


@app.delete("/api/admin/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
def delete_admin_user(
    username: str,
    current_user: TokenPayload = Depends(get_current_user),
) -> Response:
    require_admin_user(current_user)
    if username.strip() == current_user.username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own admin account")

    config = clone_config(load_config())
    try:
        delete_user_record(
            config=config,
            username=username,
            current_organization=current_user.organization,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    persist_config(config)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    token = payload.token
    if not token:
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    claims = decode_token(token)
    organization = str(claims.organization).strip()
    username = str(claims.username).strip()
    role = str(claims.role).strip().lower()

    if not organization or not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing user context")
    if not payload.prompt.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty")

    org_signature = compute_org_signature(organization)
    vector_store = load_vector_store_for_org(organization, org_signature)
    if vector_store is None:
        return ChatResponse(
            answer="Your organisation's knowledge base is currently empty. Please contact HR.",
            sources=[], username=username, organization=organization,
            role=role, model=MODEL_NAME, status="empty_index",
        )

    embeddings = load_embeddings()
    llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=SecretStr(os.getenv("GROQ_API_KEY", "")))
    memory = get_or_create_memory(username, organization)

    # Query rewriting — contextualise follow-up questions
    rewritten = rewrite_query(payload.prompt.strip(), memory, llm)

    # Hybrid retrieval — apply feedback-weighted source reputation
    t0 = time.perf_counter()
    source_weights = get_feedback_store().get_source_weights(organization)
    retriever = build_hybrid_retriever(vector_store, embeddings, source_weights=source_weights)
    result: RetrievalResult = retriever.retrieve(rewritten)
    docs = result.final_docs
    confidence = result.confidence
    total_ms = (time.perf_counter() - t0) * 1000

    if not docs:
        return ChatResponse(
            answer="I do not have that information in the current HR files. Please contact HR.",
            sources=[], username=username, organization=organization,
            role=role, model=MODEL_NAME, confidence=0.0,
            confidence_label="low", status="no_results",
        )

    conflict = detect_conflict(docs)
    context = "\n\n---\n\n".join(
        f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in docs
    )
    messages_list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {rewritten}\n\nContext:\n{context}"),
    ]
    # Measure TTFT via streaming
    t_llm = time.perf_counter()
    ttft_ms = 0.0
    answer_parts: List[str] = []
    try:
        first = True
        for chunk in llm.stream(messages_list):
            if first:
                ttft_ms = (time.perf_counter() - t_llm) * 1000
                first = False
            answer_parts.append(chunk.content)
        answer = "".join(answer_parts).strip() or "I do not have that information in the current HR files. Please contact HR."
    except Exception:
        try:
            llm_result = llm.invoke(messages_list)
            ttft_ms = (time.perf_counter() - t_llm) * 1000
            answer = llm_result.content.strip() or "I do not have that information in the current HR files. Please contact HR."
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate response") from exc

    memory.chat_memory.add_user_message(payload.prompt.strip())
    memory.chat_memory.add_ai_message(answer)

    sources = [serialize_source_document(doc) for doc in docs]
    source_names = [s.source for s in sources]
    total_ms = (time.perf_counter() - t0) * 1000

    # Build justification records for the API response
    rank_changes = result.rank_changes
    justification = [
        {
            "rank": i + 1,
            "source": Path(str(docs[i].metadata.get("source", "unknown"))).name,
            "snippet": docs[i].page_content[:400].strip(),
            "score": round(result.final_scores[i], 4) if i < len(result.final_scores) else 0.0,
            "rank_change": rank_changes[i] if i < len(rank_changes) else 0,
        }
        for i in range(len(docs))
    ]

    write_audit_log(
        username=username, org_id=organization,
        query=payload.prompt.strip(), rewritten_query=rewritten,
        answer=answer, confidence=confidence,
        sources=source_names, latency_ms=total_ms,
        conflict_detected=conflict,
    )

    query_id = f"{organization}:{username}:{int(t0 * 1000)}"

    return ChatResponse(
        answer=answer, sources=sources, username=username,
        organization=organization, role=role, model=MODEL_NAME,
        confidence=round(confidence, 4),
        confidence_label=confidence_label(confidence),
        conflict_detected=conflict,
        latency_ms=round(total_ms, 1),
        ttft_ms=round(ttft_ms, 1),
        justification=justification,
        query_id=query_id,
    )


@app.post("/api/admin/upload", response_model=UploadResponse)
async def admin_upload(
    organization: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    current_user: TokenPayload = Depends(get_current_user),
) -> UploadResponse:
    require_admin_user(current_user)

    target_org = (organization or current_user.organization or "").strip()
    if not target_org:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization is required")
    if target_org != current_user.organization:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins can only upload to their own organisation")

    target_dir = DATA_DIR / target_org
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for upload in files:
        filename = sanitize_filename(upload.filename or "uploaded_document")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".csv"}:
            continue
        destination = target_dir / filename
        # Archive existing file before overwriting (document versioning)
        archive_file_if_exists(destination)
        content = await upload.read()
        destination.write_bytes(content)
        saved_files.append(filename)

    if not saved_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No supported files uploaded")

    rebuild_organization_index(target_dir, load_embeddings())
    load_vector_store_for_org.cache_clear()
    conversation_memories.pop(get_memory_key(current_user.username, target_org), None)

    # Read drift summary if available (produced by ingest)
    drift_summary: Optional[str] = None
    drift = load_drift_report(target_org, DATA_DIR)
    if drift is not None:
        drift_summary = drift.summary

    return UploadResponse(
        organization=target_org,
        saved_files=saved_files,
        rebuilt=True,
        message=f"Rebuilt {target_org} index with {len(saved_files)} file(s).",
        drift_summary=drift_summary,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

@app.post("/api/feedback", status_code=status.HTTP_204_NO_CONTENT)
def submit_feedback(
    payload: FeedbackRequest,
    current_user: TokenPayload = Depends(get_current_user),
) -> None:
    """Record thumbs-up / thumbs-down on an answer.

    The feedback is appended to logs/feedback_log.jsonl and invalidates
    the source-weight cache for the user's organisation so that the next
    query picks up updated reputation scores.
    """
    _rate_limiter.check(current_user.organization)
    get_feedback_store().log_feedback(
        organization=current_user.organization,
        query=payload.query,
        rating=payload.rating,
        sources=payload.sources,
        confidence=payload.confidence,
        username=current_user.username,
    )


# ---------------------------------------------------------------------------
# SSE streaming chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events streaming variant of /api/chat.

    Emits:
      data: {"type": "meta", "confidence": ..., "sources": [...], ...}  (first)
      data: {"type": "token", "content": "..."}                         (per token)
      data: {"type": "ttft", "ttft_ms": ...}                            (after first token)
      data: {"type": "done", "latency_ms": ..., "query_id": "..."}      (last)
      data: {"type": "error", "detail": "..."}                          (on failure)
    """
    token = payload.token
    if not token:
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
    if not token:
        async def _err():
            yield f'data: {{"type": "error", "detail": "Missing token"}}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    try:
        claims = decode_token(token)
    except HTTPException as exc:
        async def _err2():
            yield f'data: {{"type": "error", "detail": "{exc.detail}"}}\n\n'
        return StreamingResponse(_err2(), media_type="text/event-stream")

    organization = str(claims.organization).strip()
    username = str(claims.username).strip()
    role = str(claims.role).strip().lower()

    async def event_stream() -> AsyncIterator[str]:
        import asyncio
        try:
            _rate_limiter.check(organization)
        except HTTPException as exc:
            yield f'data: {{"type": "error", "detail": "{exc.detail}"}}\n\n'
            return

        org_signature = compute_org_signature(organization)
        vector_store = load_vector_store_for_org(organization, org_signature)
        if vector_store is None:
            yield 'data: {"type": "error", "detail": "Knowledge base is empty"}\n\n'
            return

        embeddings = load_embeddings()
        llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=SecretStr(os.getenv("GROQ_API_KEY", "")))
        memory = get_or_create_memory(username, organization)
        rewritten = rewrite_query(payload.prompt.strip(), memory, llm)

        source_weights = get_feedback_store().get_source_weights(organization)
        retriever = build_hybrid_retriever(vector_store, embeddings, source_weights=source_weights)
        result: RetrievalResult = retriever.retrieve(rewritten)
        docs = result.final_docs
        confidence = result.confidence

        if not docs:
            yield 'data: {"type": "error", "detail": "No relevant documents found"}\n\n'
            return

        sources = [serialize_source_document(d) for d in docs]
        conflict = detect_conflict(docs)

        # Emit metadata first so the frontend can render confidence / sources
        meta = {
            "type": "meta",
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label(confidence),
            "conflict_detected": conflict,
            "sources": [s.model_dump() for s in sources],
        }
        yield f"data: {json.dumps(meta)}\n\n"

        context = "\n\n---\n\n".join(
            f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
            for d in docs
        )
        messages_list = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {rewritten}\n\nContext:\n{context}"),
        ]

        t0 = time.perf_counter()
        ttft_ms = 0.0
        answer_parts: List[str] = []
        first_token = True

        try:
            for chunk in llm.stream(messages_list):
                token_text = chunk.content
                if token_text:
                    if first_token:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        first_token = False
                        yield f'data: {{"type": "ttft", "ttft_ms": {ttft_ms:.1f}}}\n\n'
                    answer_parts.append(token_text)
                    safe = token_text.replace('"', '\\"').replace("\n", "\\n")
                    yield f'data: {{"type": "token", "content": "{safe}"}}\n\n'
                    await asyncio.sleep(0)  # yield control to event loop
        except Exception as exc:
            yield f'data: {{"type": "error", "detail": "Stream failed: {exc}"}}\n\n'
            return

        answer = "".join(answer_parts).strip()
        total_ms = (time.perf_counter() - t0) * 1000
        query_id = f"{organization}:{username}:{int(t0 * 1000)}"

        memory.chat_memory.add_user_message(payload.prompt.strip())
        memory.chat_memory.add_ai_message(answer)
        write_audit_log(
            username=username, org_id=organization,
            query=payload.prompt.strip(), rewritten_query=rewritten,
            answer=answer, confidence=confidence,
            sources=[s.source for s in sources],
            latency_ms=total_ms, conflict_detected=conflict,
        )

        done = {
            "type": "done",
            "latency_ms": round(total_ms, 1),
            "ttft_ms": round(ttft_ms, 1),
            "query_id": query_id,
        }
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Drift report endpoint
# ---------------------------------------------------------------------------

@app.get("/api/admin/drift-report")
def get_drift_report(
    current_user: TokenPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the most recent semantic drift report for the admin's org.

    Returns 404 if no drift report exists (e.g. no document has been
    re-uploaded yet).
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    drift = load_drift_report(current_user.organization, DATA_DIR)
    if drift is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No drift report available. Upload a replacement document to generate one.",
        )
    from dataclasses import asdict
    return asdict(drift)


@app.on_event("startup")
def startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
