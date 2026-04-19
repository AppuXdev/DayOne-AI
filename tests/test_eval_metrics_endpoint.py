from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

import main


def test_load_latest_eval_abstention_metrics(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "release_tag": "v0.8.3",
        "git_commit": "abc12345deadbeef",
        "summaries": [
            {
                "mode": "reranker_on",
                "abstention_precision": 0.9,
                "abstention_recall": 0.8,
                "abstention_f1": 0.847,
                "false_abstentions": 1,
                "false_abstention_rate": 0.1,
            },
            {
                "mode": "reranker_off",
                "abstention_precision": 0.7,
                "abstention_recall": 0.6,
                "abstention_f1": 0.646,
                "false_abstentions": 2,
                "false_abstention_rate": 0.2,
            },
        ]
    }
    eval_file = tmp_path / "eval_results.json"
    eval_file.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(main, "ROOT_DIR", tmp_path, raising=True)

    metrics = main._load_latest_eval_abstention_metrics()
    assert metrics.source_file == "eval_results.json"
    assert "reranker_on" in metrics.modes
    assert metrics.modes["reranker_on"].abstention_precision == 0.9
    assert metrics.modes["reranker_off"].false_abstentions == 2
    assert metrics.release_tag == "v0.8.3"
    assert metrics.git_commit == "abc12345deadbeef"


def test_load_recent_eval_abstention_metrics_sorted_and_limited(tmp_path: Path, monkeypatch) -> None:
    payload_old = {
        "summaries": [
            {
                "mode": "reranker_on",
                "abstention_precision": 0.5,
                "abstention_recall": 0.5,
                "abstention_f1": 0.5,
                "false_abstentions": 3,
                "false_abstention_rate": 0.3,
            }
        ]
    }
    payload_new = {
        "summaries": [
            {
                "mode": "reranker_on",
                "abstention_precision": 0.8,
                "abstention_recall": 0.9,
                "abstention_f1": 0.847,
                "false_abstentions": 1,
                "false_abstention_rate": 0.1,
            }
        ]
    }

    old_file = tmp_path / "eval_results.json"
    new_file = tmp_path / "eval_pgvector.json"
    old_file.write_text(json.dumps(payload_old), encoding="utf-8")
    new_file.write_text(json.dumps(payload_new), encoding="utf-8")

    old_mtime = 1_700_000_000
    new_mtime = 1_800_000_000
    old_file.touch()
    new_file.touch()
    import os

    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(new_file, (new_mtime, new_mtime))

    monkeypatch.setattr(main, "ROOT_DIR", tmp_path, raising=True)

    history = main._load_recent_eval_abstention_metrics(limit=2)
    assert len(history.items) == 2
    assert history.items[0].source_file == "eval_pgvector.json"
    assert history.items[1].source_file == "eval_results.json"

    latest_only = main._load_recent_eval_abstention_metrics(limit=1)
    assert len(latest_only.items) == 1
    assert latest_only.items[0].source_file == "eval_pgvector.json"


def test_eval_abstention_endpoint_requires_admin() -> None:
    user = main.TokenPayload(
        sub="john_doe",
        username="john_doe",
        organization="org_acme",
        tenant_id="tenant-1",
        role="employee",
    )
    try:
        main.get_admin_eval_abstention_metrics(current_user=user)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_eval_abstention_history_endpoint_requires_admin() -> None:
    user = main.TokenPayload(
        sub="john_doe",
        username="john_doe",
        organization="org_acme",
        tenant_id="tenant-1",
        role="employee",
    )
    try:
        main.get_admin_eval_abstention_metrics_history(limit=5, current_user=user)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_eval_artifact_endpoint_requires_admin() -> None:
    user = main.TokenPayload(
        sub="john_doe",
        username="john_doe",
        organization="org_acme",
        tenant_id="tenant-1",
        role="employee",
    )
    try:
        main.get_admin_eval_artifact(source_file="eval_results.json", current_user=user)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_eval_artifact_endpoint_returns_artifact(tmp_path: Path, monkeypatch) -> None:
    payload = {"summaries": [{"mode": "reranker_on", "abstention_f1": 0.8}]}
    eval_file = tmp_path / "eval_results.json"
    eval_file.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(main, "ROOT_DIR", tmp_path, raising=True)

    admin = main.TokenPayload(
        sub="admin",
        username="admin",
        organization="org_acme",
        tenant_id="tenant-1",
        role="admin",
    )
    response = main.get_admin_eval_artifact(source_file="eval_results.json", current_user=admin)
    assert response["source_file"] == "eval_results.json"
    assert "artifact" in response
    assert response["artifact"]["summaries"][0]["mode"] == "reranker_on"


def test_eval_artifact_endpoint_rejects_invalid_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "ROOT_DIR", tmp_path, raising=True)
    admin = main.TokenPayload(
        sub="admin",
        username="admin",
        organization="org_acme",
        tenant_id="tenant-1",
        role="admin",
    )
    try:
        main.get_admin_eval_artifact(source_file="../secrets.json", current_user=admin)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 422
