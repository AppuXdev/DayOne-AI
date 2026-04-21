"""DB-backed auth helpers for tenant-aware login.

Postgres is the single source of truth for auth after cutover.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from bcrypt import checkpw
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = "postgresql+psycopg://dayone:dayone@127.0.0.1:5432/dayone"


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
    return engine


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
