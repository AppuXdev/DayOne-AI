"""DayOne AI: Multi-tenant HR onboarding RAG SaaS (Streamlit)."""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from pydantic import SecretStr

from chat_memory import ConversationHistory
from ingest import rebuild_organization_index
from retriever import HybridRetriever, RetrievalResult, USE_RERANKER, CONF_LOW, confidence_label
from user_admin import (
    ROLE_ADMIN,
    ROLE_EMPLOYEE,
    clone_config,
    create_user_record,
    load_app_config,
    save_app_config,
    serialize_user,
    update_user_record,
    delete_user_record,
)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MASCOT_PATH = ASSETS_DIR / "mascot.png"
MODEL_NAME = os.getenv("DAYONE_GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGS_DIR.mkdir(exist_ok=True)

SUGGESTED_PROMPTS = [
    "What are the PTO rules?",
    "How does health insurance enrollment work?",
    "What is the onboarding timeline for new hires?",
]

SYSTEM_PROMPT = (
    "You are DayOne AI, a professional HR onboarding assistant. "
    "Answer ONLY from the retrieved context provided. "
    "If the answer is not in the context, say exactly: "
    "'I do not have that information in the current HR files. Please contact HR.' "
    "Do not invent, infer, or extrapolate beyond the retrieved text. "
    "If the retrieved context contains conflicting information from different "
    "documents, explicitly flag the conflict before answering."
)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
    )


def configure_page(authenticated: bool) -> None:
    st.set_page_config(
        page_title="DayOne AI",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    sidebar_css = ""
    if not authenticated:
        sidebar_css = """
        [data-testid="stSidebar"] { display: none; }
        """

    # Inject Inter font from Google Fonts
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <style>
            /* ── Global font ── */
            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif !important;
            }}

            /* ── Animated gradient background ── */
            @keyframes gradientDrift {{
                0%   {{ background-position: 0% 50%; }}
                50%  {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            .stApp {{
                background: linear-gradient(-45deg, #020617, #0b1323, #0c1a30, #060f1e);
                background-size: 400% 400%;
                animation: gradientDrift 14s ease infinite;
                color: #e2e8f0;
            }}

            /* Radial accent overlays */
            .stApp::before {{
                content: '';
                position: fixed;
                inset: 0;
                background:
                    radial-gradient(circle at 15% 25%, rgba(56, 189, 248, 0.1) 0%, transparent 40%),
                    radial-gradient(circle at 85% 75%, rgba(14, 165, 233, 0.07) 0%, transparent 40%);
                pointer-events: none;
                z-index: 0;
            }}

            #MainMenu, footer, header {{ visibility: hidden; }}
            {sidebar_css}

            /* ── Login layout ── */
            .login-wrap {{
                min-height: 72vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            /* ── Login card with glowing border ── */
            .login-card {{
                width: min(520px, 92vw);
                border-radius: 24px;
                border: 1px solid transparent;
                background: rgba(15, 23, 42, 0.85);
                padding: 2rem 2rem 1.6rem;
                box-shadow:
                    0 0 0 1px rgba(56, 189, 248, 0.15),
                    0 0 40px rgba(56, 189, 248, 0.08),
                    0 25px 60px rgba(2, 6, 23, 0.6);
                backdrop-filter: blur(20px);
                text-align: center;
            }}

            /* ── D1 monogram ── */
            .login-monogram {{
                width: 56px;
                height: 56px;
                border-radius: 16px;
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(56, 189, 248, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.25rem;
                font-size: 1.25rem;
                font-weight: 700;
                color: #38bdf8;
                letter-spacing: -0.04em;
                box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
            }}

            .login-title {{
                margin: 0 0 0.3rem 0;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.04em;
                color: #f8fafc;
            }}

            .login-subtitle {{
                margin: 0;
                color: #64748b;
                font-size: 0.9rem;
            }}

            .helper {{
                color: #64748b;
                font-size: 0.875rem;
            }}

            /* ── Welcome / zero-state card ── */
            .welcome-card {{
                border: 1px solid rgba(56, 189, 248, 0.12);
                border-radius: 20px;
                background: rgba(15, 23, 42, 0.75);
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }}

            /* ── Sidebar user card ── */
            .sidebar-user-card {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.75rem;
                border-radius: 14px;
                border: 1px solid rgba(51, 65, 85, 0.6);
                background: rgba(15, 23, 42, 0.7);
                margin-bottom: 0.75rem;
            }}

            .sidebar-avatar {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: rgba(56, 189, 248, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.35);
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.9rem;
                color: #38bdf8;
                flex-shrink: 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "memory": ConversationHistory(),
        "pending_prompt": None,
        "current_org": None,
        "current_username": None,
        "kb_missing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_config() -> dict:
    return load_app_config(CONFIG_PATH)


def persist_config(config: dict) -> None:
    save_app_config(CONFIG_PATH, config)


def require_groq_api_key() -> None:
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.error(
            "Missing GROQ_API_KEY. Add it to your .env file as GROQ_API_KEY=<your_key>, then restart DayOne AI."
        )
        st.stop()


def clear_conversation_memory() -> None:
    st.session_state.messages = []
    memory = st.session_state.get("memory")
    if hasattr(memory, "clear"):
        memory.chat_memory.clear()


def clear_session_on_logout(*_args: Any, **_kwargs: Any) -> None:
    clear_conversation_memory()
    st.session_state.clear()


def reset_invalid_auth_state() -> None:
    clear_conversation_memory()
    st.session_state.current_org = None
    st.session_state.current_username = None


def compute_org_signature(org_id: str) -> str:
    org_dir = DATA_DIR / org_id
    if not org_dir.exists():
        return "missing"

    signatures: List[str] = []
    for file_path in sorted(org_dir.rglob("*.pdf")) + sorted(org_dir.rglob("*.csv")):
        stat = file_path.stat()
        signatures.append(f"{file_path.relative_to(org_dir)}:{stat.st_mtime_ns}:{stat.st_size}")

    return "|".join(signatures) if signatures else "empty"


@st.cache_resource(show_spinner=False)
def load_vector_store_for_org(org_id: str, org_signature: str) -> Optional[FAISS]:
    if org_signature in {"missing", "empty"}:
        return None

    org_store = VECTOR_STORE_DIR / org_id
    try:
        return FAISS.load_local(
            str(org_store),
            load_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def build_hybrid_retriever(vector_store: FAISS) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,
        embeddings=load_embeddings(),
        use_reranker=USE_RERANKER,
    )


def rewrite_query(query: str, memory: ConversationHistory) -> str:
    """Rewrite a follow-up query into a standalone question using chat history.

    Skipped if there is no prior conversation (avoids unnecessary LLM call).
    Only the last two exchanges are used to keep the rewriter prompt short.
    """
    messages = memory.chat_memory.messages
    if len(messages) < 2:
        return query  # First message — no rewriting needed

    recent = messages[-4:]  # Last 2 user+assistant pairs
    history = "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {m.content}"
        for i, m in enumerate(recent)
    )
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Follow-up question: {query}\n\n"
        "Rewrite the follow-up as a fully self-contained question that can be "
        "understood without any prior context. Output ONLY the rewritten question."
    )
    try:
        result = get_llm().invoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query  # Graceful degradation


def detect_conflict(docs: List[Any]) -> bool:
    """Return True if retrieved docs come from ≥2 distinct source files.

    This is a necessary (not sufficient) condition for conflicting policies.
    The LLM system prompt instructs it to flag actual conflicts in the answer.
    """
    sources = {Path(str(d.metadata.get("source", ""))).name for d in docs}
    return len(sources) >= 2


def write_audit_log(
    username: str,
    org_id: str,
    query: str,
    rewritten_query: str,
    answer: str,
    confidence: float,
    sources: List[str],
    latency_ms: float,
    conflict_detected: bool,
) -> None:
    """Append one query record to logs/query_log.jsonl."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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


def citation_lines(source_documents: Sequence[Any]) -> List[str]:
    lines: List[str] = []
    seen = set()
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source_name = Path(str(metadata.get("source", "unknown"))).name
        page = metadata.get("page")
        row = metadata.get("row")
        if page is not None:
            label = f"{source_name} (Page {int(page) + 1})" if str(page).isdigit() else f"{source_name} (Page {page})"
        elif row is not None:
            label = f"{source_name} (Row {int(row) + 1})" if str(row).isdigit() else f"{source_name} (Row {row})"
        else:
            label = source_name
        if label not in seen:
            seen.add(label)
            lines.append(label)
    return lines


def _build_justification(result: RetrievalResult, sources: List[str]) -> List[dict]:
    """Build serialisable justification records from a RetrievalResult.

    Each record contains the chunk snippet, source, score, and rank-change
    so the UI can display explainable context for every answer.
    """
    rank_changes = result.rank_changes
    records = []
    for i, (doc, score) in enumerate(zip(result.final_docs, result.final_scores)):
        meta = getattr(doc, "metadata", {}) or {}
        source_name = Path(str(meta.get("source", "unknown"))).name
        page = meta.get("page")
        row = meta.get("row")
        if page is not None:
            loc = f"Page {int(page) + 1}" if str(page).isdigit() else f"Page {page}"
        elif row is not None:
            loc = f"Row {int(row) + 1}" if str(row).isdigit() else f"Row {row}"
        else:
            loc = ""
        records.append({
            "rank": i + 1,
            "source": source_name,
            "location": loc,
            "snippet": doc.page_content[:400].strip(),
            "score": round(score, 4),
            "rank_change": rank_changes[i] if i < len(rank_changes) else 0,
        })
    return records


def run_rag_query(
    vector_store: Optional[FAISS],
    prompt: str,
    memory: ConversationHistory,
    username: str,
    org_id: str,
) -> tuple[str, List[str], float, bool, List[dict]]:
    """Run hybrid retrieval + LLM generation.

    Returns: (answer, sources, confidence, conflict_detected, justification)
    """
    if vector_store is None:
        return (
            "Your organisation's knowledge base is currently empty. Please contact HR.",
            [], 0.0, False, [],
        )

    rewritten = rewrite_query(prompt, memory)
    retriever = build_hybrid_retriever(vector_store)

    with st.spinner("Searching policies..."):
        result: RetrievalResult = retriever.retrieve(rewritten)

    if not result.final_docs:
        return (
            "I do not have that information in the current HR files. Please contact HR.",
            [], 0.0, False, [],
        )

    conflict = detect_conflict(result.final_docs)
    context = "\n\n---\n\n".join(
        f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in result.final_docs
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {rewritten}\n\nContext:\n{context}"),
    ]
    try:
        llm_result = get_llm().invoke(messages)
        answer = llm_result.content.strip()
    except Exception:
        return (
            "I ran into a temporary retrieval issue. Please try again or contact HR support.",
            [], 0.0, False, [],
        )

    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(answer)

    sources = citation_lines(result.final_docs)
    justification = _build_justification(result, sources)
    write_audit_log(
        username=username, org_id=org_id, query=prompt,
        rewritten_query=rewritten, answer=answer,
        confidence=result.confidence, sources=sources,
        latency_ms=result.latency_ms, conflict_detected=conflict,
    )
    return answer, sources, result.confidence, conflict, justification


def render_login_hero() -> None:
    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-card">
                <div class="login-monogram">D1</div>
                <h1 class="login-title">DayOne AI</h1>
                <p class="login-subtitle">Secure multi-tenant HR onboarding assistant</p>
                <p class="helper" style="margin-top:0.35rem;">Sign in to access your organization's policy knowledge base.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_employee_sidebar(authenticator: stauth.Authenticate, org_id: str) -> None:
    # User identity card
    username_display = st.session_state.get("name") or st.session_state.get("username", "User")
    initial = (username_display[0] if username_display else "U").upper()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-user-card">
            <div class="sidebar-avatar">{initial}</div>
            <div style="min-width:0;">
                <p style="margin:0;font-size:0.875rem;font-weight:600;color:#f1f5f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(username_display)}</p>
                <p style="margin:0;font-size:0.72rem;color:#64748b;">Employee</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Org: **{org_id}**")

    if MASCOT_PATH.exists():
        st.sidebar.image(str(MASCOT_PATH), use_container_width=True)
    else:
        st.sidebar.info("Mascot asset not found at assets/mascot.png")

    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        clear_conversation_memory()
        st.rerun()

    authenticator.logout("Sign Out", "sidebar", callback=clear_session_on_logout)


def _users_for_org(config: dict, org_id: str) -> List[dict]:
    credentials = config.get("credentials", {})
    usernames = credentials.get("usernames", {})
    if not isinstance(usernames, dict):
        return []
    scoped = [
        serialize_user(username, record)
        for username, record in usernames.items()
        if str(record.get("organization", "")).strip() == org_id
    ]
    return sorted(scoped, key=lambda item: item["username"].lower())


def render_admin_user_management(config: dict, current_username: str, org_id: str) -> None:
    st.markdown("### User Management")
    st.caption("Create, update, and remove users for this organization without editing config files.")

    users = _users_for_org(config, org_id)
    if users:
        st.dataframe(
            [
                {
                    "Username": user["username"],
                    "Name": user["name"] or "—",
                    "Email": user["email"] or "—",
                    "Role": user["role"],
                }
                for user in users
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No users are configured for this organization yet.")

    create_col, update_col = st.columns(2)

    with create_col:
        with st.form("create-user-form", clear_on_submit=True):
            st.markdown("#### Add User")
            new_username = st.text_input("Username", help="3-64 chars using letters, numbers, ., _, or -")
            new_name = st.text_input("Full name")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role", options=[ROLE_EMPLOYEE, ROLE_ADMIN], index=0)
            new_password = st.text_input("Temporary password", type="password", help="Minimum 8 characters")
            create_submitted = st.form_submit_button("Create User", use_container_width=True)

        if create_submitted:
            updated_config = clone_config(config)
            try:
                create_user_record(
                    config=updated_config,
                    username=new_username,
                    password=new_password,
                    organization=org_id,
                    role=new_role,
                    name=new_name,
                    email=new_email,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                persist_config(updated_config)
                st.success(f"Created user `{new_username.strip()}`.")
                st.rerun()

    with update_col:
        user_options = [user["username"] for user in users]
        selected_username = st.selectbox(
            "Select user",
            options=user_options,
            index=0 if user_options else None,
            placeholder="Choose a user",
        )

        selected_user = next((user for user in users if user["username"] == selected_username), None)
        if selected_user:
            with st.form("update-user-form"):
                st.markdown("#### Edit User")
                edit_name = st.text_input("Full name", value=selected_user["name"])
                edit_email = st.text_input("Email", value=selected_user["email"])
                role_index = 0 if selected_user["role"] == ROLE_EMPLOYEE else 1
                edit_role = st.selectbox("Role", options=[ROLE_EMPLOYEE, ROLE_ADMIN], index=role_index)
                edit_password = st.text_input(
                    "New password",
                    type="password",
                    help="Leave blank to keep the existing password",
                )
                save_user = st.form_submit_button("Save Changes", use_container_width=True)

            if save_user:
                updated_config = clone_config(config)
                try:
                    update_user_record(
                        config=updated_config,
                        username=selected_username,
                        current_organization=org_id,
                        name=edit_name,
                        email=edit_email,
                        role=edit_role,
                        password=edit_password or None,
                    )
                except (PermissionError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    persist_config(updated_config)
                    st.success(f"Updated user `{selected_username}`.")
                    st.rerun()

            delete_disabled = selected_username == current_username
            delete_help = "You cannot delete the account you are currently using." if delete_disabled else None
            if st.button("Delete User", use_container_width=True, disabled=delete_disabled, help=delete_help):
                updated_config = clone_config(config)
                try:
                    delete_user_record(
                        config=updated_config,
                        username=selected_username,
                        current_organization=org_id,
                    )
                except (PermissionError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    persist_config(updated_config)
                    st.success(f"Deleted user `{selected_username}`.")
                    st.rerun()


def render_admin_portal(config: dict, username: str, org_id: str) -> None:
    st.markdown("## Administration Portal")
    st.caption("Upload policy documents for your tenant and rebuild the index.")

    upload_org = st.selectbox("Organization", options=[org_id], disabled=True)
    uploaded_files = st.file_uploader(
        "Upload policy files (.pdf, .csv)",
        type=["pdf", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Save Files and Rebuild Index", width="stretch"):
        if not uploaded_files:
            st.warning("Please choose at least one file.")
            return

        target_dir = DATA_DIR / upload_org
        target_dir.mkdir(parents=True, exist_ok=True)

        saved: List[str] = []
        for upload in uploaded_files:
            filename = Path(upload.name).name
            if Path(filename).suffix.lower() not in {".pdf", ".csv"}:
                continue
            destination = target_dir / filename
            destination.write_bytes(upload.getbuffer())
            saved.append(filename)

        if not saved:
            st.error("No supported files were uploaded.")
            return

        with st.status("Rebuilding tenant knowledge index…", expanded=True) as rebuild_status:
            st.write("📄 Parsing and chunking documents…")
            rebuild_organization_index(target_dir, load_embeddings())
            load_vector_store_for_org.clear()
            rebuild_status.update(
                label=f"✅ Index rebuilt for **{upload_org}** ({len(saved)} file(s))",
                state="complete",
                expanded=False,
            )

        st.session_state.chain = None
        st.success(f"Rebuilt {upload_org} index with {len(saved)} file(s).")
        st.caption("Saved: " + ", ".join(saved))
        st.caption(f"Updated by: {username}")

    st.divider()
    render_admin_user_management(config, username, org_id)


SUGGESTION_ICONS = ["🕐", "🏥", "📅"]


def render_zero_state() -> None:
    st.markdown(
        """
        <div class="welcome-card">
            <h3 style="margin:0; color:#f8fafc; font-size:1.25rem; font-weight:700; letter-spacing:-0.02em;">Welcome to DayOne AI</h3>
            <p class="helper" style="margin-top:0.4rem;">
                Ask about onboarding, benefits, leave, and HR policies specific to your organization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for col, prompt, icon in zip(cols, SUGGESTED_PROMPTS, SUGGESTION_ICONS):
        with col:
            if st.button(f"{icon} {prompt}", use_container_width=True):
                st.session_state.pending_prompt = prompt


def _render_justification(justification: List[dict], used_reranker: bool) -> None:
    """Render the Answer Justification expander — the 'explainable RAG' layer."""
    if not justification:
        return
    score_label = "Reranker score" if used_reranker else "BM25 score"
    with st.expander(
        f"🔍 Answer Justification — {len(justification)} retrieved chunk(s) "
        f"| {score_label} shown",
        expanded=False,
    ):
        if used_reranker:
            st.caption(
                f"Retrieved {12} candidates via BM25+FAISS→RRF, "
                f"then cross-encoder reranked to top {len(justification)}. "
                "\u2191N = promoted N positions by reranker."
            )
        else:
            st.caption("Reranker OFF — BM25+FAISS→RRF fusion only. Score = BM25 score.")

        for rec in justification:
            change = rec["rank_change"]
            arrow = f"↑{change}" if change > 0 else ("↓{abs(change)}" if change < 0 else "—")
            loc = f" · {rec['location']}" if rec["location"] else ""
            st.markdown(
                f"**#{rec['rank']} — `{rec['source']}`{loc}** "
                f"&nbsp;&nbsp; `{score_label}: {rec['score']:.3f}` "
                f"&nbsp; `Rank change: {arrow}`"
            )
            st.markdown(
                f"> {rec['snippet'].replace(chr(10), ' ')[:350]}…"
                if len(rec["snippet"]) > 350 else f"> {rec['snippet']}"
            )
            st.divider()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant":
                conf = message.get("confidence", 0.0)
                conflict = message.get("conflict_detected", False)
                sources = message.get("sources", [])
                justification = message.get("justification", [])
                used_reranker = message.get("used_reranker", USE_RERANKER)

                label = confidence_label(conf)
                colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪")
                st.caption(f"{colour} Confidence: {label} ({conf:.0%})  |  Reranker: {'ON' if used_reranker else 'OFF'}  |  Chunks: {len(justification)}")

                if conflict:
                    st.warning(
                        "⚠️ Retrieved context spans multiple documents. "
                        "Conflicting policies may exist — verify with HR before acting.",
                        icon="⚠️",
                    )
                if conf < CONF_LOW and conf > 0:
                    st.info(
                        "ℹ️ Low retrieval confidence. This answer may be incomplete — "
                        "please cross-check with HR."
                    )

                _render_justification(justification, used_reranker)

                if sources:
                    with st.expander("📎 View Sources"):
                        for source in sources:
                            st.markdown(f"- {html.escape(source)}")


def render_employee_chat(authenticator: stauth.Authenticate, org_id: str) -> None:
    render_employee_sidebar(authenticator, org_id)

    st.markdown("## HR Policy Assistant")
    if st.session_state.kb_missing:
        st.warning("Your organisation's knowledge base is currently empty. Please contact HR.")

    if st.session_state.messages:
        render_chat_history()
    else:
        render_zero_state()

    typed_prompt = st.chat_input("Ask an HR policy question...")
    active_prompt = st.session_state.get("pending_prompt") or typed_prompt
    st.session_state.pending_prompt = None

    if not active_prompt:
        return

    normalized_prompt = active_prompt.strip()
    if not normalized_prompt:
        return

    username = str(st.session_state.get("current_username", "unknown"))
    st.session_state.messages.append({"role": "user", "content": normalized_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(normalized_prompt)

    answer, sources, confidence, conflict, justification = run_rag_query(
        st.session_state.get("vector_store"),
        normalized_prompt,
        st.session_state.memory,
        username,
        org_id,
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence,
        "conflict_detected": conflict,
        "justification": justification,
        "used_reranker": USE_RERANKER,
    })
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
        label = confidence_label(confidence)
        colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪")
        st.caption(f"{colour} Confidence: {label} ({confidence:.0%})  |  Reranker: {'ON' if USE_RERANKER else 'OFF'}  |  Chunks: {len(justification)}")
        if conflict:
            st.warning("⚠️ Retrieved context spans multiple documents — verify with HR.", icon="⚠️")
        if confidence < CONF_LOW and confidence > 0:
            st.info("ℹ️ Low retrieval confidence — please cross-check with HR.")
        _render_justification(justification, USE_RERANKER)
        if sources:
            with st.expander("📎 View Sources"):
                for source in sources:
                    st.markdown(f"- {html.escape(source)}")


def main() -> None:
    load_dotenv()
    initialize_state()

    auth_status = bool(st.session_state.get("authentication_status"))
    configure_page(authenticated=auth_status)
    require_groq_api_key()

    config = load_config()
    credentials_root = config.get("credentials", {})
    users = credentials_root.get("usernames", {})
    authenticator = stauth.Authenticate(
        credentials=credentials_root,
        cookie_name=config.get("cookie", {}).get("name", "dayone_ai_auth"),
        cookie_key=config.get("cookie", {}).get("key", "change-this-key"),
        cookie_expiry_days=config.get("cookie", {}).get("expiry_days", 30),
        preauthorized=config.get("preauthorized"),
        auto_hash=False,
    )

    if st.session_state.get("authentication_status") is not True:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            with st.container(border=True):
                st.markdown("### DayOne AI")
                st.caption("Secure multi-tenant HR onboarding assistant")
                authenticator.login(fields={"Form name": "Sign in"})

    if st.session_state.get("authentication_status") is True:
        username = str(st.session_state.get("username", "")).strip()
        name = st.session_state.get("name")

        user_info = users.get(username, {})
        org_id = str(user_info.get("organization", "")).strip()
        user_role = str(user_info.get("role", "employee") or "employee").strip().lower()

        if not username or not user_info:
            reset_invalid_auth_state()
            st.error("Authenticated user mapping is invalid. Contact admin.")
            st.stop()

        if not org_id:
            reset_invalid_auth_state()
            st.error("No organization is mapped to this account.")
            st.stop()

        if st.session_state.current_org != org_id or st.session_state.current_username != username:
            clear_conversation_memory()
            st.session_state.current_org = org_id
            st.session_state.current_username = username

        org_signature = compute_org_signature(org_id)
        vector_store = load_vector_store_for_org(org_id, org_signature)
        st.session_state.vector_store = vector_store
        st.session_state.kb_missing = vector_store is None

        st.caption(f"Signed in as {name or username} | Org: {org_id} | Role: {user_role}")

        if user_role == ROLE_ADMIN:
            authenticator.logout("Sign Out", "main", callback=clear_session_on_logout)
            render_admin_portal(config, username, org_id)
            return
        elif user_role == ROLE_EMPLOYEE:
            render_employee_chat(authenticator, org_id)
            return

        reset_invalid_auth_state()
        st.error("Unsupported role configuration.")
        st.stop()

    if st.session_state.get("authentication_status") is False:
        reset_invalid_auth_state()
        st.error("Invalid username or password.")
        st.stop()

    st.stop()


if __name__ == "__main__":
    main()
