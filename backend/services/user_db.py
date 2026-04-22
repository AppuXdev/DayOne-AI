"""PostgreSQL user CRUD for tenant-scoped admin operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bcrypt import gensalt, hashpw
from sqlalchemy import text

from backend.services.auth_db import require_engine

ROLE_ADMIN = "admin"
ROLE_EMPLOYEE = "employee"
VALID_ROLES = {ROLE_ADMIN, ROLE_EMPLOYEE}


def _normalize_role(role: str) -> str:
    normalized = (role or ROLE_EMPLOYEE).strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError("Role must be either 'admin' or 'employee'")
    return normalized


def _hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    return hashpw(password.encode("utf-8"), gensalt()).decode("utf-8")


def _tenant_id_for_org(conn, organization: str) -> Optional[str]:
    row = conn.execute(
        text("SELECT id FROM tenants WHERE lower(name) = lower(:name)"),
        {"name": organization.strip()},
    ).mappings().first()
    return str(row["id"]) if row else None


def create_organization(organization_name: str) -> str:
    """Create a new tenant (organization). Returns the tenant_id."""
    engine = require_engine()
    with engine.begin() as conn:
        # Check if exists
        exists = _tenant_id_for_org(conn, organization_name)
        if exists:
            raise ValueError(f"Organization '{organization_name}' already exists")

        row = conn.execute(
            text(
                """
                INSERT INTO tenants (name)
                VALUES (:name)
                RETURNING id
                """
            ),
            {"name": organization_name.strip()},
        ).mappings().first()
        return str(row["id"])


def get_org_stats(organization: str) -> Dict[str, Any]:
    """Return counts of users and documents for an organization."""
    engine = require_engine()
    with engine.connect() as conn:
        tenant_row = conn.execute(
            text("SELECT id, name FROM tenants WHERE lower(name) = lower(:name)"),
            {"name": organization.strip()},
        ).mappings().first()
        
        if not tenant_row:
            raise ValueError(f"Organization '{organization}' not found")
            
        tenant_id = tenant_row["id"]
        
        user_count = conn.execute(
            text("SELECT COUNT(*) FROM users WHERE tenant_id = :tenant_id"),
            {"tenant_id": tenant_id},
        ).scalar() or 0
        
        doc_count = conn.execute(
            text("SELECT COUNT(*) FROM documents WHERE tenant_id = :tenant_id AND status != 'deleted'"),
            {"tenant_id": tenant_id},
        ).scalar() or 0
        
    return {
        "id": str(tenant_id),
        "name": str(tenant_row["name"]),
        "user_count": int(user_count),
        "document_count": int(doc_count),
    }


def list_users_for_org(organization: str) -> List[Dict[str, Any]]:
    engine = require_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    u.username,
                    COALESCE(u.name, '') AS name,
                    COALESCE(u.email, '') AS email,
                    t.name AS organization,
                    u.role
                FROM users u
                JOIN tenants t ON t.id = u.tenant_id
                WHERE lower(t.name) = lower(:organization)
                ORDER BY lower(u.username)
                """
            ),
            {"organization": organization.strip()},
        ).mappings().all()

    return [
        {
            "username": str(r["username"]),
            "name": str(r["name"]),
            "email": str(r["email"]),
            "organization": str(r["organization"]),
            "role": str(r["role"]),
        }
        for r in rows
    ]


def create_user(
    *,
    organization: str,
    username: str,
    password: str,
    role: str,
    name: str = "",
    email: str = "",
) -> Dict[str, Any]:
    engine = require_engine()
    normalized_username = username.strip().lower()
    if len(normalized_username) < 3:
        raise ValueError("Username must be at least 3 characters long")

    with engine.begin() as conn:
        tenant_id = _tenant_id_for_org(conn, organization)
        if tenant_id is None:
            raise ValueError(f"Organization '{organization}' not found")
        
        exists = conn.execute(
            text(
                """
                SELECT 1
                FROM users
                WHERE tenant_id = :tenant_id
                  AND lower(username) = lower(:username)
                LIMIT 1
                """
            ),
            {"tenant_id": tenant_id, "username": normalized_username},
        ).first()
        if exists:
            raise ValueError(f"User '{normalized_username}' already exists")

        conn.execute(
            text(
                """
                INSERT INTO users (username, password_hash, tenant_id, role, name, email)
                VALUES (:username, :password_hash, :tenant_id, :role, :name, :email)
                """
            ),
            {
                "username": normalized_username,
                "password_hash": _hash_password(password),
                "tenant_id": tenant_id,
                "role": _normalize_role(role),
                "name": name.strip(),
                "email": email.strip(),
            },
        )

    return {
        "username": normalized_username,
        "name": name.strip(),
        "email": email.strip(),
        "organization": organization.strip(),
        "role": _normalize_role(role),
    }


def update_user(
    *,
    organization: str,
    username: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    engine = require_engine()
    normalized_username = username.strip().lower()

    with engine.begin() as conn:
        tenant_id = _tenant_id_for_org(conn, organization)
        if tenant_id is None:
            raise ValueError(f"Organisation '{organization}' not found")

        row = conn.execute(
            text(
                """
                SELECT id, username, role, COALESCE(name, '') AS name, COALESCE(email, '') AS email
                FROM users
                WHERE tenant_id = :tenant_id
                  AND lower(username) = lower(:username)
                LIMIT 1
                """
            ),
            {"tenant_id": tenant_id, "username": normalized_username},
        ).mappings().first()
        if row is None:
            raise ValueError(f"User '{normalized_username}' was not found")

        updates: List[str] = []
        params: Dict[str, Any] = {"id": str(row["id"])}

        if name is not None:
            updates.append("name = :name")
            params["name"] = name.strip()
        if email is not None:
            updates.append("email = :email")
            params["email"] = email.strip()
        if role is not None:
            updates.append("role = :role")
            params["role"] = _normalize_role(role)
        if password is not None and password.strip():
            updates.append("password_hash = :password_hash")
            params["password_hash"] = _hash_password(password)

        if updates:
            conn.execute(
                text(f"UPDATE users SET {', '.join(updates)} WHERE id = :id"),
                params,
            )

        refreshed = conn.execute(
            text(
                """
                SELECT u.username, COALESCE(u.name, '') AS name, COALESCE(u.email, '') AS email, u.role, t.name AS organization
                FROM users u
                JOIN tenants t ON t.id = u.tenant_id
                WHERE u.id = :id
                """
            ),
            {"id": str(row["id"])}
        ).mappings().first()

    return {
        "username": str(refreshed["username"]),
        "name": str(refreshed["name"]),
        "email": str(refreshed["email"]),
        "organization": str(refreshed["organization"]),
        "role": str(refreshed["role"]),
    }


def delete_user(*, organization: str, username: str) -> None:
    engine = require_engine()
    normalized_username = username.strip().lower()

    with engine.begin() as conn:
        tenant_id = _tenant_id_for_org(conn, organization)
        if tenant_id is None:
            raise ValueError(f"Organisation '{organization}' not found")

        result = conn.execute(
            text(
                """
                DELETE FROM users
                WHERE tenant_id = :tenant_id
                  AND lower(username) = lower(:username)
                """
            ),
            {"tenant_id": tenant_id, "username": normalized_username},
        )
        if result.rowcount == 0:
            raise ValueError(f"User '{normalized_username}' was not found")
