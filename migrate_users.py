"""One-time migration: config.yaml users -> PostgreSQL.

Usage:
    python migrate_users.py

Requires:
    DATABASE_URL environment variable
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def _load_config_users(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    creds = raw.get("credentials", {}) if isinstance(raw, dict) else {}
    users = creds.get("usernames", {}) if isinstance(creds, dict) else {}
    if not isinstance(users, dict):
        return {}
    return users


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    users = _load_config_users(CONFIG_PATH)
    if not users:
        print("No users found in config.yaml")
        return

    engine = create_engine(database_url, pool_pre_ping=True)

    migrated = 0
    seen_keys: set[tuple[str, str]] = set()
    with engine.begin() as conn:
        for username, record in users.items():
            if not isinstance(record, dict):
                continue

            organization = str(record.get("organization", "")).strip()
            role = str(record.get("role", "employee") or "employee").strip().lower()
            password_hash = str(record.get("password", "")).strip()
            name = str(record.get("name", "")).strip()
            email = str(record.get("email", "")).strip()

            normalized_username = username.strip().lower()
            normalized_org = organization.strip().lower()

            if not normalized_username or not normalized_org or not password_hash:
                continue
            if role not in {"admin", "employee"}:
                role = "employee"

            dedupe_key = (normalized_org, normalized_username)
            if dedupe_key in seen_keys:
                print(f"Skipping duplicate config user entry: {organization}/{username}")
                continue
            seen_keys.add(dedupe_key)

            tenant_row = conn.execute(
                text(
                    """
                    INSERT INTO tenants (name)
                    VALUES (:name)
                    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """
                ),
                {"name": organization},
            ).mappings().first()

            tenant_id = str(tenant_row["id"])

            conn.execute(
                text(
                    """
                    INSERT INTO users (username, password_hash, tenant_id, role, name, email)
                    VALUES (:username, :password_hash, :tenant_id, :role, :name, :email)
                    ON CONFLICT (tenant_id, username) DO UPDATE
                    SET
                        password_hash = EXCLUDED.password_hash,
                        role = EXCLUDED.role,
                        name = EXCLUDED.name,
                        email = EXCLUDED.email
                    """
                ),
                {
                    "username": normalized_username,
                    "password_hash": password_hash,
                    "tenant_id": tenant_id,
                    "role": role,
                    "name": name,
                    "email": email,
                },
            )
            migrated += 1

    print(f"Migrated {migrated} user(s) from config.yaml to PostgreSQL")


if __name__ == "__main__":
    main()
