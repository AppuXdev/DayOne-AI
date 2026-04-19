"""Watch DayOne AI source folders and auto-rebuild tenant indexes.

Run this script in a separate terminal alongside the Streamlit app:

    python auto_ingest.py

It watches data/org_*/ folders for PDF and CSV changes and rebuilds only the
affected organization's pgvector embeddings.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import ingest


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


class OrganizationChangeHandler(FileSystemEventHandler):
    """Debounced handler that refreshes only the changed organization embeddings."""

    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self.debounce_seconds = debounce_seconds
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self.embeddings = ingest.build_embeddings()

    def _organization_from_path(self, path: str) -> Optional[str]:
        file_path = Path(path)
        for parent in file_path.parents:
            if parent == DATA_DIR:
                return None
            if parent.parent == DATA_DIR and parent.name.startswith("org_"):
                return parent.name
        return None

    def _schedule_rebuild(self, org_id: str) -> None:
        with self._lock:
            existing_timer = self._timers.get(org_id)
            if existing_timer is not None:
                existing_timer.cancel()

            timer = threading.Timer(self.debounce_seconds, self._rebuild, args=(org_id,))
            timer.daemon = True
            self._timers[org_id] = timer
            timer.start()

    def _rebuild(self, org_id: str) -> None:
        org_dir = DATA_DIR / org_id
        if not org_dir.exists():
            print(f"[{org_id}] source folder removed; skipping embedding refresh.")
            return

        print(f"[{org_id}] refreshing embeddings...")
        ingest.rebuild_organization_index(org_dir, self.embeddings)

    def _maybe_handle(self, event_path: str) -> None:
        if not event_path.lower().endswith((".csv", ".pdf")):
            return

        org_id = self._organization_from_path(event_path)
        if org_id:
            self._schedule_rebuild(org_id)

    def on_created(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._maybe_handle(event.src_path)

    def on_modified(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._maybe_handle(event.src_path)

    def on_moved(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._maybe_handle(getattr(event, "dest_path", event.src_path))

    def on_deleted(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._maybe_handle(event.src_path)


def main() -> None:
    """Start the filesystem watcher and keep it running."""

    load_dotenv()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    handler = OrganizationChangeHandler()
    observer = Observer()
    observer.schedule(handler, str(DATA_DIR), recursive=True)
    observer.start()

    print(f"Watching {DATA_DIR} for new or changed PDFs/CSVs...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
