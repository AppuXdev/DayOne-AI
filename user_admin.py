"""Shared helpers for DayOne AI user and config management."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, Optional

import yaml
from bcrypt import gensalt, hashpw

ROLE_ADMIN = "admin"
ROLE_EMPLOYEE = "employee"
VALID_ROLES = {ROLE_ADMIN, ROLE_EMPLOYEE}
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,64}$")


def load_app_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    # streamlit-authenticator v0.4.x expects these keys on each user record.
    usernames = (
        config.get("credentials", {})
        .get("usernames", {})
        if isinstance(config, dict)
        else {}
    )
    if isinstance(usernames, dict):
        for record in usernames.values():
            if isinstance(record, dict):
                record.setdefault("failed_login_attempts", 0)
                record.setdefault("logged_in", False)

    return config


def save_app_config(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    path.write_text(rendered, encoding="utf-8")


def get_user_map(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    credentials = config.setdefault("credentials", {})
    usernames = credentials.setdefault("usernames", {})
    if not isinstance(usernames, dict):
        raise ValueError("credentials.usernames must be a mapping")
    return usernames


def clone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(config)


def normalize_role(role: str) -> str:
    normalized = (role or ROLE_EMPLOYEE).strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError("Role must be either 'admin' or 'employee'")
    return normalized


def normalize_username(username: str) -> str:
    normalized = username.strip()
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError("Username must be 3-64 chars and use letters, numbers, ., _, or -")
    return normalized


def normalize_organization(organization: str) -> str:
    normalized = organization.strip()
    if not normalized:
        raise ValueError("Organization is required")
    return normalized


def hash_password_value(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    return hashpw(password.encode("utf-8"), gensalt()).decode("utf-8")


def serialize_user(username: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": username,
        "name": str(record.get("name", "")).strip(),
        "email": str(record.get("email", "")).strip(),
        "organization": str(record.get("organization", "")).strip(),
        "role": normalize_role(str(record.get("role", ROLE_EMPLOYEE))),
    }


def create_user_record(
    *,
    config: Dict[str, Any],
    username: str,
    password: str,
    organization: str,
    role: str,
    name: str = "",
    email: str = "",
) -> Dict[str, Any]:
    usernames = get_user_map(config)
    normalized_username = normalize_username(username)
    if normalized_username in usernames:
        raise ValueError(f"User '{normalized_username}' already exists")

    usernames[normalized_username] = {
        "email": email.strip(),
        "name": name.strip(),
        "password": hash_password_value(password),
        "organization": normalize_organization(organization),
        "role": normalize_role(role),
    }
    return serialize_user(normalized_username, usernames[normalized_username])


def update_user_record(
    *,
    config: Dict[str, Any],
    username: str,
    current_organization: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    usernames = get_user_map(config)
    normalized_username = normalize_username(username)
    record = usernames.get(normalized_username)
    if record is None:
        raise ValueError(f"User '{normalized_username}' was not found")

    user_org = normalize_organization(str(record.get("organization", "")))
    if user_org != normalize_organization(current_organization):
        raise PermissionError("You can only manage users in your own organization")

    if name is not None:
        record["name"] = name.strip()
    if email is not None:
        record["email"] = email.strip()
    if role is not None:
        record["role"] = normalize_role(role)
    if password is not None and password.strip():
        record["password"] = hash_password_value(password)

    return serialize_user(normalized_username, record)


def delete_user_record(
    *,
    config: Dict[str, Any],
    username: str,
    current_organization: str,
) -> None:
    usernames = get_user_map(config)
    normalized_username = normalize_username(username)
    record = usernames.get(normalized_username)
    if record is None:
        raise ValueError(f"User '{normalized_username}' was not found")

    user_org = normalize_organization(str(record.get("organization", "")))
    if user_org != normalize_organization(current_organization):
        raise PermissionError("You can only manage users in your own organization")

    del usernames[normalized_username]
