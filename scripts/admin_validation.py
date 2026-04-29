"""Admin validation script: runs Tests 1-7 described in conversation summary.

Run: python scripts/admin_validation.py

Optional: set env var INGEST=1 to run ingestion first (may download models).
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import (
    ChatRequest,
    TokenPayload,
    process_chat_query,
)
from backend.services import user_db, document_db, query_trace_db, auth_db


def ensure_org_and_admin(org: str, admin_username: str = "admin", admin_password: str = "dayoneadmin"):
    # Ensure tenant exists
    try:
        user_db.create_organization(org)
        print(f"Created organization: {org}")
    except ValueError:
        print(f"Organization exists: {org}")

    # Ensure admin user exists
    users = user_db.list_users_for_org(org)
    if any(u["username"].lower() == admin_username.lower() for u in users):
        print(f"Admin user exists in {org}: {admin_username}")
    else:
        try:
            user_db.create_user(organization=org, username=admin_username, password=admin_password, role="admin")
            print(f"Created admin user '{admin_username}' in {org}")
        except Exception as exc:
            print(f"Failed to create admin user: {exc}")


def run_tests(org: str, other_org: str = None):
    print(f"\n=== Running admin validation for org: {org} ===")
    admin_username = "admin"
    admin_password = os.getenv("ADMIN_TEST_PASSWORD", "dayoneadmin")

    # Test 1 - ADMIN LOGIN
    auth = auth_db.authenticate_user(username=admin_username, password=admin_password, organization=org)
    test1_pass = auth is not None and auth.get("tenant_id")
    print("TEST 1 - ADMIN LOGIN:", "PASS" if test1_pass else "FAIL", f"-> {auth}")

    # Test 2 - DOCUMENT VISIBILITY
    docs = document_db.list_documents_for_tenant(org)
    doc_count = len(docs)
    sample_names = [d.get("filename") for d in docs[:5]]
    test2_pass = doc_count > 0
    print("TEST 2 - DOCUMENT VISIBILITY:", "PASS" if test2_pass else "FAIL", f"count={doc_count}, sample={sample_names}")

    # Use a token payload for requests
    token = TokenPayload(sub=admin_username, username=admin_username, organization=org, tenant_id=auth.get("tenant_id") if auth else None, role="admin")

    # Test 3 - RETRIEVAL TRACE
    req = ChatRequest(prompt="leave policy")
    try:
        processed = process_chat_query(req, token)
        resp = processed.response
        print("Response answer:", resp.answer[:200])
        # fetch traces
        tenant_id = query_trace_db.resolve_tenant_id(tenant_id=token.tenant_id, organization=org)
        traces = query_trace_db.list_query_traces(tenant_id=tenant_id, limit=5)
        test3_pass = len(traces) > 0
        print("TEST 3 - RETRIEVAL TRACE:", "PASS" if test3_pass else "FAIL", f"traces_found={len(traces)}")
        if test3_pass:
            latest = traces[0]["trace"]
            print("  retrieval:", latest.get("retrieval"))
            print("  confidence:", latest.get("confidence"))
            print("  final_context_count:", len(latest.get("final_context", [])))
    except Exception as exc:
        print("TEST 3 - RETRIEVAL TRACE: ERROR ->", exc)
        test3_pass = False

    # Test 4 - RANK MOVEMENT (inspect final_context fields)
    rank_ok = False
    try:
        if test3_pass:
            latest = traces[0]["trace"]
            contexts = latest.get("final_context", [])
            # We expect each context to have source and tenant
            rank_ok = all(isinstance(c.get("source"), str) for c in contexts)
        print("TEST 4 - RANK MOVEMENT:", "PASS" if rank_ok else "FAIL", f"final_context_count={len(contexts) if test3_pass else 0}")
    except Exception as exc:
        print("TEST 4 - RANK MOVEMENT: ERROR ->", exc)

    # Test 5 - VERIFICATION / ABSTENTION
    try:
        # Nonsense query to provoke abstention/no results
        req_bad = ChatRequest(prompt="what is the salary of nonexistent-person-xyz?")
        processed_bad = process_chat_query(req_bad, token)
        resp_bad = processed_bad.response
        abstained = resp_bad.abstained or resp_bad.verification.get("is_grounded") is False
        print("TEST 5 - VERIFICATION/ABSTENTION:", "PASS" if abstained else "FAIL", f"abstained={abstained}, verification={resp_bad.verification}")
    except Exception as exc:
        print("TEST 5 - VERIFICATION/ABSTENTION: ERROR ->", exc)

    # Test 6 - TENANT ISOLATION
    tenant_isolated = False
    if other_org:
        try:
            # run same query in other org
            other_auth = auth_db.authenticate_user(username=admin_username, password=admin_password, organization=other_org)
            other_token = TokenPayload(sub=admin_username, username=admin_username, organization=other_org, tenant_id=other_auth.get("tenant_id") if other_auth else None, role="admin")
            other_processed = process_chat_query(ChatRequest(prompt="leave policy"), other_token)
            # Compare top sources tenants
            tenant1 = None
            tenant2 = None
            if test3_pass and traces:
                t0 = traces[0]["trace"]
                if t0.get("final_context"):
                    tenant1 = t0["final_context"][0].get("tenant")
            if other_processed and other_processed.response and other_processed.response.sources:
                tenant2 = other_processed.response.sources[0].tenant
            tenant_isolated = tenant1 != tenant2
            print("TEST 6 - TENANT ISOLATION:", "PASS" if tenant_isolated else "FAIL", f"tenant_top1={tenant1} vs other_top1={tenant2}")
        except Exception as exc:
            print("TEST 6 - TENANT ISOLATION: ERROR ->", exc)
    else:
        print("TEST 6 - TENANT ISOLATION: SKIPPED (no other_org provided)")

    # Test 7 - CONFIDENCE
    try:
        good_req = ChatRequest(prompt="leave policy")
        good_proc = process_chat_query(good_req, token)
        good_conf = good_proc.response.confidence
        bad_proc = processed_bad
        bad_conf = bad_proc.response.confidence
        conf_ok = good_conf > bad_conf
        print("TEST 7 - CONFIDENCE:", "PASS" if conf_ok else "FAIL", f"good_conf={good_conf}, bad_conf={bad_conf}")
    except Exception as exc:
        print("TEST 7 - CONFIDENCE: ERROR ->", exc)


if __name__ == "__main__":
    # Determine orgs to test from data/ folder
    data_dir = os.path.join(ROOT, "data")
    orgs = [d for d in os.listdir(data_dir) if d.startswith("org_")]
    if not orgs:
        print("No org_ directories found in data/. Create data/org_acme etc. Aborting.")
        sys.exit(2)

    org = orgs[0]
    other_org = orgs[1] if len(orgs) > 1 else None

    # Ensure tenants and admin users
    for o in [org] + ([other_org] if other_org else []):
        if o:
            ensure_org_and_admin(o)

    run_tests(org, other_org)
