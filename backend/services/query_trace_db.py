"""PostgreSQL-backed query trace storage for uncertainty observability."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import text

from backend.services.auth_db import require_engine


def _tenant_id_for_org(conn, organization: str) -> Optional[str]:
    row = conn.execute(
        text("SELECT id FROM tenants WHERE lower(name) = lower(:name)"),
        {"name": organization.strip()},
    ).mappings().first()
    return str(row["id"]) if row else None


def resolve_tenant_id(*, tenant_id: Optional[str], organization: str) -> str:
    if tenant_id and tenant_id.strip():
        return tenant_id.strip()

    engine = require_engine()
    with engine.connect() as conn:
        resolved = _tenant_id_for_org(conn, organization)
    if not resolved:
        raise ValueError(f"Organisation '{organization}' was not found")
    return resolved


def store_query_trace(*, tenant_id: str, query: str, trace: Dict[str, Any], trace_id: Optional[str] = None) -> str:
    engine = require_engine()
    trace_id = trace_id or str(uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO query_traces (id, tenant_id, query, trace)
                VALUES (:id, CAST(:tenant_id AS uuid), :query, CAST(:trace AS jsonb))
                """
            ),
            {
                "id": trace_id,
                "tenant_id": tenant_id,
                "query": query,
                "trace": json.dumps(trace, ensure_ascii=True, default=str),
            },
        )
    return trace_id


def list_query_traces(
    *,
    tenant_id: str,
    limit: int = 50,
    query_type: Optional[str] = None,
    abstained: Optional[bool] = None,
    low_confidence: bool = False,
) -> List[Dict[str, Any]]:
    engine = require_engine()
    limit = max(1, min(int(limit or 50), 200))

    conditions = ["tenant_id = CAST(:tenant_id AS uuid)"]
    params: Dict[str, Any] = {"tenant_id": tenant_id, "limit": limit}

    if query_type:
        conditions.append("trace ->> 'query_type' = :query_type")
        params["query_type"] = query_type.strip()
    if abstained is not None:
        conditions.append("COALESCE((trace ->> 'abstained')::boolean, false) = :abstained")
        params["abstained"] = abstained
    if low_confidence:
        conditions.append("COALESCE((trace ->> 'confidence')::numeric, 0) < 0.4")

    query = f"""
        SELECT id, tenant_id, query, trace, created_at
        FROM query_traces
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT :limit
    """

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return [
        {
            "id": str(row["id"]),
            "tenant_id": str(row["tenant_id"]),
            "query": str(row["query"]),
            "trace": row["trace"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
