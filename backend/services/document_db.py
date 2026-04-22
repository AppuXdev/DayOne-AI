"""PostgreSQL document metadata lifecycle for MinIO-backed storage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from sqlalchemy import text

from backend.services.auth_db import require_engine
from backend.services.storage_minio import bucket_name

STATUS_UPLOADING = "uploading"
STATUS_PROCESSING = "processing"
STATUS_ACTIVE = "active"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"
VALID_STATUSES = {
    STATUS_UPLOADING,
    STATUS_PROCESSING,
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_DELETED,
}


def _normalize_status(status: str) -> str:
    value = (status or "").strip().lower()
    if value not in VALID_STATUSES:
        raise ValueError(f"Invalid document status: {status}")
    return value


def _tenant_id_for_org(conn, organization: str) -> str:
    row = conn.execute(
        text("SELECT id FROM tenants WHERE lower(name) = lower(:name)"),
        {"name": organization.strip()},
    ).mappings().first()
    if row is None:
        raise ValueError(f"Organization '{organization}' not found")
    return str(row["id"])


def create_document_row(
    *,
    organization: str,
    filename: str,
    object_key: str,
    uploaded_by_username: str,
    status: str = STATUS_UPLOADING,
) -> Dict[str, Any]:
    engine = require_engine()
    normalized_status = _normalize_status(status)

    with engine.begin() as conn:
        tenant_id = _tenant_id_for_org(conn, organization)

        uploader_row = conn.execute(
            text(
                """
                SELECT id
                FROM users
                WHERE tenant_id = :tenant_id
                  AND lower(username) = lower(:username)
                LIMIT 1
                """
            ),
            {"tenant_id": tenant_id, "username": uploaded_by_username.strip().lower()},
        ).mappings().first()
        uploaded_by = str(uploader_row["id"]) if uploader_row else None

        version_row = conn.execute(
            text(
                """
                SELECT COALESCE(MAX(version), 0) AS max_version
                FROM documents
                WHERE tenant_id = :tenant_id
                  AND lower(filename) = lower(:filename)
                """
            ),
            {"tenant_id": tenant_id, "filename": filename.strip()},
        ).mappings().first()
        next_version = int(version_row["max_version"] or 0) + 1

        inserted = conn.execute(
            text(
                """
                INSERT INTO documents
                    (tenant_id, file_url, filename, object_key, version, status, uploaded_by)
                VALUES
                    (:tenant_id, :file_url, :filename, :object_key, :version, :status, :uploaded_by)
                RETURNING id, filename, object_key, version, status
                """
            ),
            {
                "tenant_id": tenant_id,
                "file_url": f"minio://{bucket_name()}/{object_key}",
                "filename": filename.strip(),
                "object_key": object_key,
                "version": next_version,
                "status": normalized_status,
                "uploaded_by": uploaded_by,
            },
        ).mappings().first()

    return {
        "id": str(inserted["id"]),
        "filename": str(inserted["filename"]),
        "object_key": str(inserted["object_key"]),
        "version": int(inserted["version"]),
        "status": str(inserted["status"]),
    }


def set_documents_status(document_ids: Iterable[str], status: str, error_message: str = "") -> None:
    doc_ids = [str(d) for d in document_ids if str(d).strip()]
    if not doc_ids:
        return
    engine = require_engine()
    normalized_status = _normalize_status(status)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE documents
                SET status = :status,
                    error_message = :error_message,
                    updated_at = now()
                WHERE id = ANY(:ids)
                """
            ),
            {"status": normalized_status, "error_message": error_message[:500], "ids": doc_ids},
        )


def list_documents_for_tenant(
    organization: str,
    statuses: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    engine = require_engine()
    status_list = [s.strip().lower() for s in statuses] if statuses else []

    query = (
        """
        SELECT d.id, d.filename, d.object_key, d.version, d.status, t.name AS organization
        FROM documents d
        JOIN tenants t ON t.id = d.tenant_id
        WHERE lower(t.name) = lower(:organization)
        """
    )
    params: Dict[str, Any] = {"organization": organization.strip()}
    if status_list:
        query += " AND d.status = ANY(:statuses)"
        params["statuses"] = status_list
    query += " ORDER BY lower(d.filename), d.version DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return [
        {
            "id": str(r["id"]),
            "filename": str(r["filename"]),
            "object_key": str(r["object_key"]),
            "version": int(r["version"]),
            "status": str(r["status"]),
            "organization": str(r["organization"]),
        }
        for r in rows
    ]


def org_signature(organization: str) -> str:
    """Compute a stable cache key for tenant vector-store invalidation."""
    engine = require_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS n,
                    COALESCE(SUM(version), 0) AS v_sum,
                    COALESCE(MAX(updated_at), MAX(created_at)) AS latest
                FROM documents d
                JOIN tenants t ON t.id = d.tenant_id
                WHERE lower(t.name) = lower(:organization)
                  AND d.status = 'active'
                """
            ),
            {"organization": organization.strip()},
        ).mappings().first()

    n = int(row["n"] or 0)
    if n == 0:
        return "empty"
    v_sum = int(row["v_sum"] or 0)
    latest = row["latest"]
    latest_str = latest.isoformat() if latest is not None else "none"
    return f"{n}:{v_sum}:{latest_str}"


def list_organizations_with_documents() -> List[str]:
    engine = require_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT t.name AS organization
                FROM documents d
                JOIN tenants t ON t.id = d.tenant_id
                WHERE d.status <> 'deleted'
                ORDER BY lower(t.name)
                """
            )
        ).mappings().all()
    return [str(r["organization"]) for r in rows]
