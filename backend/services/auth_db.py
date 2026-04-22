"""DB-backed auth helpers for tenant-aware login.

Postgres is the single source of truth for auth after cutover.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Dict, Optional

from bcrypt import checkpw
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = "postgresql+psycopg://dayone:dayone@127.0.0.1:5432/dayone"
_schema_lock = threading.Lock()
_schema_ready = False


@lru_cache(maxsize=1)
def _database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip() or DEFAULT_DATABASE_URL


@lru_cache(maxsize=1)
def _engine() -> Optional[Engine]:
    url = _database_url()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True)


def is_enabled() -> bool:
    """Return True when DATABASE_URL is configured."""
    return bool(_database_url())


def require_engine() -> Engine:
    """Return a live SQLAlchemy engine or raise a runtime error.

    Runtime should fail fast when DB config is missing, since Postgres is the
    single source of truth after auth cutover.
    """
    engine = _engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is required for authentication and user management")
    ensure_schema(engine)
    return engine


def ensure_schema(engine: Optional[Engine] = None) -> None:
    """Create the core tenant/auth tables when they do not exist yet."""
    global _schema_ready
    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        db_engine = engine or _engine()
        if db_engine is None:
            raise RuntimeError("DATABASE_URL is required for authentication and user management")

        statements = [
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('admin', 'employee')),
                name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (tenant_id, username)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                file_url TEXT NOT NULL,
                object_key TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploading' CHECK (status IN ('uploading', 'processing', 'active', 'failed', 'deleted')),
                error_message TEXT NOT NULL DEFAULT '',
                uploaded_by UUID NULL REFERENCES users(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS object_key TEXT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'uploading'",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_message TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_tenants_name_lower
            ON tenants (lower(name))
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_users_tenant_username_lower
            ON users (tenant_id, lower(username))
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_tenant_filename_version
            ON documents (tenant_id, lower(filename), version)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_tenant_object_key
            ON documents (tenant_id, object_key)
            WHERE object_key IS NOT NULL
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
            """
            UPDATE users
            SET username = lower(username)
            WHERE username <> lower(username)
            """,
        ]

        with db_engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

        _schema_ready = True


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def authenticate_user(
    *,
    username: str,
    password: str,
    organization: str,
) -> Optional[Dict[str, Any]]:
    """Authenticate user from PostgreSQL.

    Returns a normalized user payload on success, else None.
    """
    engine = _engine()
    if engine is None:
        return None
    ensure_schema(engine)

    query = text(
        """
        SELECT
            u.id,
            u.username,
            u.password_hash,
            u.role,
            u.tenant_id,
            t.name AS organization
        FROM users u
        JOIN tenants t ON t.id = u.tenant_id
        WHERE lower(u.username) = lower(:username)
                    AND lower(t.name) = lower(:organization)
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "username": username,
                "organization": organization.strip(),
            },
        ).mappings().first()

    if row is None:
        return None

    if not verify_password(password, str(row["password_hash"])):
        return None

    return {
        "username": str(row["username"]).strip(),
        "organization": str(row["organization"]).strip(),
        "role": str(row["role"]).strip().lower() or "employee",
        "tenant_id": str(row["tenant_id"]),
        "id": str(row["id"]),
    }
