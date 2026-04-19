"""MinIO object storage adapter with tenant-safe key helpers."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Dict, List

from minio import Minio
from minio.error import S3Error


@lru_cache(maxsize=1)
def _settings() -> Dict[str, str]:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000").strip()
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin").strip()
    secret_key = os.getenv("MINIO_SECRET_KEY", "password").strip()
    bucket = os.getenv("MINIO_BUCKET", "dayone-docs").strip()
    secure = os.getenv("MINIO_SECURE", "0").strip() in {"1", "true", "True"}
    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
        "secure": "1" if secure else "0",
    }


@lru_cache(maxsize=1)
def _client() -> Minio:
    s = _settings()
    return Minio(
        s["endpoint"],
        access_key=s["access_key"],
        secret_key=s["secret_key"],
        secure=s["secure"] == "1",
    )


def bucket_name() -> str:
    return _settings()["bucket"]


def ensure_bucket() -> None:
    client = _client()
    b = bucket_name()
    if not client.bucket_exists(b):
        client.make_bucket(b)


def put_bytes(object_key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
    ensure_bucket()
    client = _client()
    stream = io.BytesIO(payload)
    client.put_object(
        bucket_name(),
        object_key,
        data=stream,
        length=len(payload),
        content_type=content_type,
    )


def get_bytes(object_key: str) -> bytes:
    client = _client()
    response = client.get_object(bucket_name(), object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def exists(object_key: str) -> bool:
    client = _client()
    try:
        client.stat_object(bucket_name(), object_key)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return False
        raise


def delete_object(object_key: str) -> None:
    client = _client()
    try:
        client.remove_object(bucket_name(), object_key)
    except S3Error as exc:
        if exc.code not in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise


def list_keys(prefix: str = "") -> List[str]:
    client = _client()
    objects = client.list_objects(bucket_name(), prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]
