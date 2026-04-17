"""Feedback store for DayOne AI.

Stores user thumbs-up / thumbs-down on answers and computes per-source
reputation scores that are applied by HybridRetriever to weight chunks
from historically useful (or harmful) sources.

Source reputation formula
-------------------------
    reputation(source, org) = 0.5 + 0.5 * tanh(2 * net_ratio)
    net_ratio               = (positive - negative) / max(total, 1)

Range ≈ [0.27, 0.73].  Scaled to a weight centred at 1.0:
    weight = 1.0 + (reputation - 0.5)  →  [0.77, 1.23]

The ±23% range is intentionally modest: a single piece of feedback cannot
cause a source to be completely ignored or fully dominant.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
FEEDBACK_LOG = ROOT_DIR / "logs" / "feedback_log.jsonl"


class FeedbackStore:
    """Thread-safe feedback store backed by a JSONL append-only log.

    Usage
    -----
        store = FeedbackStore()
        store.log_feedback("org_acme", "What is PTO?", "up", ["handbook.pdf"], 0.82)
        weights = store.get_source_weights("org_acme")
        # {"handbook.pdf": 1.12, "old_policy.pdf": 0.91, ...}
    """

    def __init__(self, log_path: Path = FEEDBACK_LOG) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        # Per-org weight cache: invalidated on new feedback for that org
        self._weight_cache: Dict[str, Dict[str, float]] = {}
        self._cache_valid: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        organization: str,
        query: str,
        rating: str,            # "up" | "down"
        sources: List[str],
        confidence: float,
        username: str = "",
    ) -> None:
        """Append one feedback record and invalidate weight cache for the org."""
        self.log_path.parent.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organization": organization,
            "username": username,
            "query": query,
            "rating": rating,
            "sources": sources,
            "confidence": round(confidence, 4),
        }
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            self._cache_valid[organization] = False

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_source_weights(self, organization: str) -> Dict[str, float]:
        """Return a {source_name: weight} dict for the given org.

        Weight = 1.0 → neutral (no feedback).
        Weight > 1.0 → historically useful; boosted during retrieval.
        Weight < 1.0 → historically unhelpful; down-weighted.

        The dict is recomputed only when new feedback has been logged
        since the last call (lazy cache invalidation).
        """
        with self._lock:
            if self._cache_valid.get(organization, False):
                return self._weight_cache.get(organization, {})

        weights = self._compute_weights(organization)

        with self._lock:
            self._weight_cache[organization] = weights
            self._cache_valid[organization] = True

        return weights

    def get_stats(self, organization: str) -> Dict[str, object]:
        """Summary stats for admin display."""
        weights = self.get_source_weights(organization)
        total = up = 0

        if self.log_path.exists():
            with self._lock:
                lines = self.log_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("organization") == organization:
                    total += 1
                    if record.get("rating") == "up":
                        up += 1

        return {
            "total_feedback": total,
            "up": up,
            "down": total - up,
            "source_weights": weights,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(self, organization: str) -> Dict[str, float]:
        """Scan the feedback log and compute reputation weights per source."""
        if not self.log_path.exists():
            return {}

        counts: Dict[str, Dict[str, int]] = {}  # source → {"up": n, "down": n}

        with self._lock:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("organization") != organization:
                continue

            rating = record.get("rating", "")
            for source in record.get("sources", []):
                if source not in counts:
                    counts[source] = {"up": 0, "down": 0}
                if rating == "up":
                    counts[source]["up"] += 1
                elif rating == "down":
                    counts[source]["down"] += 1

        weights: Dict[str, float] = {}
        for source, c in counts.items():
            pos, neg = c["up"], c["down"]
            total = pos + neg
            net_ratio = (pos - neg) / max(total, 1)
            # tanh maps net_ratio ∈ [-1, 1] → reputation ∈ [0.27, 0.73]
            reputation = 0.5 + 0.5 * math.tanh(2.0 * net_ratio)
            # Scale to weight centred at 1.0 → [0.77, 1.23]
            weights[source] = 1.0 + (reputation - 0.5)

        return weights


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Return the process-level FeedbackStore singleton."""
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
