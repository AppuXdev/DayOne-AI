"""Semantic drift detection for DayOne AI.

When a policy document is re-uploaded (replacing an existing version), this
module compares the old and new chunk sets at the embedding level to produce
a DriftReport that identifies:

  - Sections that changed semantically (cosine distance > threshold)
  - Sections that are brand new (no close neighbour in the old set)
  - Sections that were removed (old chunks with no close match in new set)
  - Sections that are unchanged

This turns file archiving into semantic version control — admins can see
exactly which HR policies changed, not just that a file was replaced.

Algorithm
---------
1. Embed all chunks in the old and new sets.
2. Normalise to unit vectors (cosine similarity = dot product).
3. For each new chunk, find its nearest neighbour in the old set.
4. If distance < DRIFT_THRESHOLD: unchanged; else: changed.
5. Old chunks with no close match in the new set: removed.
6. Chunks in net excess of the old set count: new (approximate).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Cosine distance above this threshold is considered a "meaningful" change.
# Set via DRIFT_DISTANCE_THRESHOLD env var (float, 0–2 range; default 0.25).
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_DISTANCE_THRESHOLD", "0.25"))
# Maximum diffs stored in the report (cap for readability).
MAX_DIFFS = 20


@dataclass
class ChunkDiff:
    """A single chunk-level difference between old and new document versions."""
    status: str          # "new" | "removed" | "changed" | "unchanged"
    old_snippet: str = ""
    new_snippet: str = ""
    distance: float = 0.0
    source: str = ""


@dataclass
class DriftReport:
    """Semantic diff summary for one document upload event."""
    organization: str
    document: str
    timestamp: str
    new_chunks: int
    removed_chunks: int
    changed_chunks: int
    unchanged_chunks: int
    diffs: List[ChunkDiff] = field(default_factory=list)
    summary: str = ""

    @property
    def has_drift(self) -> bool:
        return (self.new_chunks + self.removed_chunks + self.changed_chunks) > 0


class SemanticDriftDetector:
    """Compare two sets of LangChain Document chunks at the embedding level."""

    def __init__(
        self,
        embeddings: Any,
        threshold: float = DRIFT_THRESHOLD,
    ) -> None:
        self.embeddings = embeddings
        self.threshold = threshold

    def compare_chunk_sets(
        self,
        old_chunks: List[Any],
        new_chunks: List[Any],
        organization: str,
        document_name: str,
    ) -> DriftReport:
        """Produce a DriftReport for one document replacement.

        Parameters
        ----------
        old_chunks : chunks from the previously indexed version
        new_chunks : chunks from the freshly uploaded version
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if not old_chunks:
            # First-time upload — everything is new
            diffs = [
                ChunkDiff(
                    status="new",
                    new_snippet=c.page_content[:200],
                    source=document_name,
                )
                for c in new_chunks[:MAX_DIFFS]
            ]
            return DriftReport(
                organization=organization,
                document=document_name,
                timestamp=timestamp,
                new_chunks=len(new_chunks),
                removed_chunks=0,
                changed_chunks=0,
                unchanged_chunks=0,
                summary=f"New document: {len(new_chunks)} section(s) added.",
                diffs=diffs,
            )

        old_texts = [c.page_content for c in old_chunks]
        new_texts = [c.page_content for c in new_chunks]

        # Embed both sets — potentially slow for large documents
        old_vecs = np.array(
            self.embeddings.embed_documents(old_texts), dtype=np.float32
        )
        new_vecs = np.array(
            self.embeddings.embed_documents(new_texts), dtype=np.float32
        )

        # L2-normalise for cosine distance via dot product
        old_vecs = old_vecs / np.maximum(
            np.linalg.norm(old_vecs, axis=1, keepdims=True), 1e-8
        )
        new_vecs = new_vecs / np.maximum(
            np.linalg.norm(new_vecs, axis=1, keepdims=True), 1e-8
        )

        # Similarity matrix: shape (n_new, n_old)
        sim_matrix = new_vecs @ old_vecs.T
        dist_matrix = 1.0 - sim_matrix

        matched_old: set[int] = set()
        diffs: List[ChunkDiff] = []
        new_count = removed_count = changed_count = unchanged_count = 0

        for ni, new_chunk in enumerate(new_chunks):
            best_old_idx = int(np.argmin(dist_matrix[ni]))
            best_dist = float(dist_matrix[ni, best_old_idx])
            matched_old.add(best_old_idx)

            if best_dist < self.threshold:
                unchanged_count += 1
                status = "unchanged"
            else:
                changed_count += 1
                status = "changed"

            if len(diffs) < MAX_DIFFS and status == "changed":
                diffs.append(
                    ChunkDiff(
                        status=status,
                        old_snippet=old_chunks[best_old_idx].page_content[:200],
                        new_snippet=new_chunk.page_content[:200],
                        distance=round(best_dist, 4),
                        source=document_name,
                    )
                )

        # Old chunks without a close neighbour in the new set → removed
        for oi in range(len(old_chunks)):
            if oi not in matched_old:
                removed_count += 1
                if len(diffs) < MAX_DIFFS:
                    diffs.append(
                        ChunkDiff(
                            status="removed",
                            old_snippet=old_chunks[oi].page_content[:200],
                            source=document_name,
                        )
                    )

        # Net-new chunks: new set is larger than old set
        net_new = max(0, len(new_chunks) - len(old_chunks))
        new_count = net_new

        summary_parts: List[str] = []
        if changed_count:
            summary_parts.append(f"{changed_count} section(s) changed")
        if new_count:
            summary_parts.append(f"{new_count} section(s) added")
        if removed_count:
            summary_parts.append(f"{removed_count} section(s) removed")
        if not summary_parts:
            summary_parts.append("No significant policy changes detected")

        summary = "; ".join(summary_parts) + "."

        return DriftReport(
            organization=organization,
            document=document_name,
            timestamp=timestamp,
            new_chunks=new_count,
            removed_chunks=removed_count,
            changed_chunks=changed_count,
            unchanged_chunks=unchanged_count,
            diffs=diffs,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_drift_report(report: DriftReport, data_dir: Path) -> Path:
    """Write drift_report.json to data/<org>/ and return the path."""
    org_dir = data_dir / report.organization
    org_dir.mkdir(parents=True, exist_ok=True)
    report_path = org_dir / "drift_report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report_path


def load_drift_report(
    organization: str,
    data_dir: Path,
) -> Optional[DriftReport]:
    """Load the most recent drift report for an org, or None."""
    report_path = data_dir / organization / "drift_report.json"
    if not report_path.exists():
        return None
    try:
        raw: Dict = json.loads(report_path.read_text(encoding="utf-8"))
        diffs = [ChunkDiff(**d) for d in raw.pop("diffs", [])]
        return DriftReport(**raw, diffs=diffs)
    except Exception:
        return None
