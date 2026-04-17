"""Tenant-aware ingestion pipeline for DayOne AI (CSV/PDF -> FAISS per org).

Document versioning
-------------------
When a file that already exists in an org's data directory is uploaded again,
the old version is archived to  data/<org>/archive/<stem>.<timestamp><ext>
before the new file is written. This provides a simple audit trail and
enables rollback without requiring a database.

Trade-off: archives grow unboundedly. In production this would be backed
by object storage with a lifecycle policy. For a local demo, a manual
`rm -rf data/<org>/archive/` is sufficient.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TENANT_MAX_CHUNKS: int = int(os.getenv("TENANT_MAX_CHUNKS", "50000"))


def list_org_dirs(data_dir: Path = DATA_DIR) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("org_"))


def list_source_files(org_dir: Path) -> List[Path]:
    return sorted(list(org_dir.rglob("*.pdf")) + list(org_dir.rglob("*.csv")))


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_docs_for_file(path: Path) -> List:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".csv":
        return CSVLoader(file_path=str(path), encoding="utf-8").load()
    return []


def archive_file_if_exists(destination: Path) -> Optional[Path]:
    """Move an existing file to the archive/ subfolder with a timestamp suffix.

    Returns the archive path if a file was moved, else None.
    """
    if not destination.exists():
        return None
    archive_dir = destination.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{destination.stem}.{stamp}{destination.suffix}"
    destination.rename(archive_path)
    return archive_path


# ---------------------------------------------------------------------------
# Incremental ingestion helpers
# ---------------------------------------------------------------------------

HASH_REGISTRY_FILE = ".file_hashes.json"
CHUNK_CACHE_FILE = ".chunk_cache.pkl"


def compute_file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hash_registry(org_dir: Path) -> Dict[str, str]:
    """Load the {filename: sha256} registry from disk."""
    registry_path = org_dir / HASH_REGISTRY_FILE
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_hash_registry(org_dir: Path, registry: Dict[str, str]) -> None:
    (org_dir / HASH_REGISTRY_FILE).write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


def load_chunk_cache(org_dir: Path) -> Dict[str, List]:
    """Load the {filename: [chunk, ...]} cache from disk."""
    cache_path = org_dir / CHUNK_CACHE_FILE
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                return pickle.load(fh)  # noqa: S301
        except Exception:
            pass
    return {}


def save_chunk_cache(org_dir: Path, cache: Dict[str, List]) -> None:
    with (org_dir / CHUNK_CACHE_FILE).open("wb") as fh:
        pickle.dump(cache, fh)


def rebuild_organization_index(
    org_dir: Path,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    out_dir: Optional[Path] = None,
    incremental: bool = True,
) -> None:
    """Rebuild the FAISS vector index for a single organisation.

    Incremental mode (default ON)
    ------------------------------
    Tracks a SHA-256 hash per source file in `.file_hashes.json`.
    Chunks for unchanged files are loaded from `.chunk_cache.pkl` instead
    of being re-parsed. Changed or new files are fully re-chunked.

    Trade-off: the FAISS index is always rebuilt from all chunks (true
    incremental FAISS merge requires careful docstore management and is
    deferred). The benefit is that chunking — typically the slow step for
    large PDFs — is skipped for unchanged files.

    Chunking parameters
    -------------------
    chunk_size=1000, chunk_overlap=200
      Chosen after empirical testing on HR PDFs. See README for rationale.
    """
    embeddings = embeddings or build_embeddings()
    files = list_source_files(org_dir)
    # Exclude archive and hidden files from ingestion
    files = [f for f in files if "archive" not in f.parts and not f.name.startswith(".")]
    org_name = org_dir.name
    out_dir = out_dir or (VECTOR_STORE_DIR / org_name)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    if not files:
        print(f"[{org_name}] skipped: no source files.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks: List = []

    # Load existing hash registry and chunk cache
    hash_registry = load_hash_registry(org_dir) if incremental else {}
    chunk_cache = load_chunk_cache(org_dir) if incremental else {}
    new_registry: Dict[str, str] = {}
    new_cache: Dict[str, List] = {}
    reused = 0
    rechunked = 0

    for file_path in files:
        fname = file_path.name
        current_hash = compute_file_hash(file_path)
        new_registry[fname] = current_hash

        if incremental and hash_registry.get(fname) == current_hash and fname in chunk_cache:
            # File unchanged — reuse cached chunks
            cached = chunk_cache[fname]
            all_chunks.extend(cached)
            new_cache[fname] = cached
            reused += 1
            print(f"[{org_name}] {fname}: unchanged — using cached chunks ({len(cached)} chunks)")
            continue

        # File is new or modified — re-chunk
        try:
            docs = load_docs_for_file(file_path)
        except Exception as exc:
            print(f"[{org_name}] failed loading {fname}: {exc}")
            continue

        for doc in docs:
            doc.metadata["tenant"] = org_name
            doc.metadata["source"] = str(file_path)
            doc.metadata["ingested_at"] = datetime.now().isoformat()
            doc.metadata["file_hash"] = current_hash

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        new_cache[fname] = chunks
        rechunked += 1
        print(f"[{org_name}] {fname}: {'modified' if fname in hash_registry else 'new'} — {len(chunks)} chunks")

    if not all_chunks:
        print(f"[{org_name}] skipped: no parseable chunks.")
        return

    # Enforce per-tenant chunk limit
    if len(all_chunks) > TENANT_MAX_CHUNKS:
        raise RuntimeError(
            f"[{org_name}] chunk count {len(all_chunks)} exceeds TENANT_MAX_CHUNKS={TENANT_MAX_CHUNKS}. "
            "Remove old documents or increase the limit via the TENANT_MAX_CHUNKS env var."
        )

    print(f"[{org_name}] ingestion summary: {reused} file(s) cached, {rechunked} file(s) re-chunked, {len(all_chunks)} total chunks")

    # Semantic drift detection — run only when at least one file was re-chunked
    # and a previous chunk cache existed (i.e. this is an update, not a first ingest).
    if rechunked > 0 and chunk_cache:
        try:
            from drift import SemanticDriftDetector, save_drift_report  # noqa: PLC0415
            detector = SemanticDriftDetector(embeddings)
            # Detect drift for each rechunked file that existed previously
            for fname, new_file_chunks in new_cache.items():
                if fname in chunk_cache and fname in new_cache:
                    old_file_chunks = chunk_cache[fname]
                    if old_file_chunks:  # skip truly new files
                        report = detector.compare_chunk_sets(
                            old_chunks=old_file_chunks,
                            new_chunks=new_file_chunks,
                            organization=org_name,
                            document_name=fname,
                        )
                        save_drift_report(report, DATA_DIR)
                        print(f"[{org_name}] drift report: {report.summary}")
        except Exception as exc:
            # Drift detection is non-critical — log and continue
            print(f"[{org_name}] drift detection skipped: {exc}")

    # FAISS Flat index (IndexFlatL2) — exact NN search. See README for trade-offs.
    store = FAISS.from_documents(all_chunks, embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))

    # Persist updated registry and cache
    if incremental:
        save_hash_registry(org_dir, new_registry)
        save_chunk_cache(org_dir, new_cache)

    print(f"[{org_name}] index built — {store.index.ntotal} vectors → {out_dir}")


def build_all_organization_indexes() -> None:
    load_dotenv()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings()
    org_dirs = list_org_dirs(DATA_DIR)

    if not org_dirs:
        raise RuntimeError("No tenant folders found. Create folders like data/org_acme first.")

    for org_dir in org_dirs:
        rebuild_organization_index(org_dir, embeddings)


def main() -> None:
    build_all_organization_indexes()


if __name__ == "__main__":
    main()
