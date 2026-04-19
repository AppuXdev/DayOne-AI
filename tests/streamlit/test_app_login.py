# Diagnosis summary (ticket bdf7c32d-d6be-475d-bb2d-48b82acac6fe)
# - streamlit-authenticator version: 0.4.2
# - bcrypt.checkpw(b"password123", hash): admin_acme=False (before fix), john_doe=True (before fix)
# - config.yaml mutation check after launching Streamlit briefly: changed=False
# - Fixes applied: auto_hash=False in authenticator, user metadata defaults backfill,
#   admin_acme hash aligned to password123, duplicate pre-authorized key removed.

"""Regression tests for Streamlit login.

Manual verification:
1. Run `streamlit run app.py`.
2. Login as employee using `john_doe` / `password123` and verify chat page loads.
3. Login as admin using `admin_acme` / `password123` and verify admin portal loads.
"""

from __future__ import annotations

from typing import Iterable
from unittest.mock import MagicMock

import pytest
import dotenv
import langchain_groq
import langchain_huggingface

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def _extract_values(elements: Iterable[object]) -> list[str]:
    values: list[str] = []
    for el in elements:
        value = getattr(el, "value", "")
        values.append(str(value))
    return values


def _apply_runtime_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYONE_USE_RERANKER", "0")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False, raising=True)

    # Patch constructors used inside app.py when loaded from file.
    monkeypatch.setattr(
        langchain_huggingface,
        "HuggingFaceEmbeddings",
        MagicMock(return_value=MagicMock(name="embeddings_ctor")),
        raising=True,
    )
    monkeypatch.setattr(
        langchain_groq,
        "ChatGroq",
        MagicMock(return_value=MagicMock(name="llm_ctor")),
        raising=True,
    )

    import app

    # Required by ticket: keep patches inline in this test module.
    monkeypatch.setattr(app, "load_embeddings", MagicMock(return_value=MagicMock(name="embeddings")), raising=True)
    monkeypatch.setattr(app, "get_llm", MagicMock(return_value=MagicMock(name="llm")), raising=True)

    # Keep login tests isolated from model loading.


def _submit_login(at: AppTest, username: str, password: str) -> AppTest:
    at.run()
    assert len(at.text_input) >= 2
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)

    if len(at.button) > 0:
        at.button[0].click()
    else:
        raise AssertionError("Login submit control was not rendered")

    at.run()
    return at


def test_streamlit_login_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_runtime_patches(monkeypatch)

    at = AppTest.from_file("app.py", default_timeout=20)
    at = _submit_login(at, "john_doe", "password123")

    assert at.session_state["authentication_status"] is True
    captions = _extract_values(at.caption)
    assert any("Org:" in value for value in captions)


def test_streamlit_login_wrong_password(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_runtime_patches(monkeypatch)

    at = AppTest.from_file("app.py", default_timeout=20)
    at = _submit_login(at, "john_doe", "wrong-password")

    assert at.session_state["authentication_status"] is False
    errors = _extract_values(at.error)
    assert any("Invalid" in value for value in errors)


def test_streamlit_login_missing_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False, raising=True)

    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()

    errors = _extract_values(at.error)
    assert any("Missing GROQ_API_KEY" in value for value in errors)
