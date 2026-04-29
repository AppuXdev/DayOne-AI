"""PostgreSQL pgvector embedding persistence for tenant-scoped retrieval."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sqlalchemy import text

from backend.services.auth_db import require_engine

EMBEDDING_DIM: int = int(os.getenv("DAYONE_EMBEDDING_DIM", "384"))


def _tenant_id_for_org(conn, organization: str) -> str:
    """Return the tenant UUID for *organization*, creating the row if absent."""
    org_name = organization.strip()
    if not org_name:
        raise ValueError("organization name must not be empty")

    row = conn.execute(
        text("SELECT id FROM tenants WHERE lower(name) = lower(:name)"),
        {"name": org_name},
    ).mappings().first()

    if row is None:
        row = conn.execute(
            text(
                """
                INSERT INTO tenants (name)
                VALUES (:name)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """
            ),
            {"name": org_name},
        ).mappings().first()

        if row is None:
            raise RuntimeError(f"Tenant upsert for '{org_name}' returned no id — check DB permissions.")
            
    return str(row["id"])


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _normalize(vec: Iterable[float]) -> List[float]:
    arr = np.array(list(vec), dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {arr.shape[0]}")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Zero-norm embedding encountered")
    arr = arr / norm
    return arr.tolist()


def replace_tenant_embeddings(
    *,
    organization: str,
    chunks: List[Any],
    embedding_model: Any,
) -> Tuple[str, int]:
    """Replace all embeddings for a tenant in one transaction.

    Maintains a fully consistent tenant snapshot for pgvector retrieval.
    """
    engine = require_engine()

    if not chunks:
        with engine.begin() as conn:
            tenant_id = _tenant_id_for_org(conn, organization)
            conn.execute(text("DELETE FROM embeddings WHERE tenant_id = :tenant_id"), {"tenant_id": tenant_id})
        return tenant_id, 0

    texts = [str(c.page_content) for c in chunks]
    vectors = embedding_model.embed_documents(texts)

    rows: List[Dict[str, Any]] = []
    for c, vec in zip(chunks, vectors):
        metadata = getattr(c, "metadata", {}) or {}
        rows.append(
            {
                "chunk_text": str(c.page_content),
                "embedding": _vector_literal(_normalize(vec)),
                "metadata": json.dumps(metadata, ensure_ascii=True, default=str),
            }
        )

    with engine.begin() as conn:
        tenant_id = _tenant_id_for_org(conn, organization)
        conn.execute(text("DELETE FROM embeddings WHERE tenant_id = :tenant_id"), {"tenant_id": tenant_id})

        stmt = text(
            """
            INSERT INTO embeddings (tenant_id, chunk_text, embedding, metadata)
            VALUES (:tenant_id, :chunk_text, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
            """
        )

        batch_size = 250
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            payload = [{"tenant_id": tenant_id, **item} for item in batch]
            conn.execute(stmt, payload)

    return tenant_id, len(rows)
