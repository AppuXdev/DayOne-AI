"""Abstention policy for uncertainty-aware answer handling.

This module is intentionally model-agnostic and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AbstentionDecision:
    abstained: bool
    reason: Optional[str]


@dataclass(frozen=True)
class VerificationSummary:
    is_grounded: bool
    conflict_detected: bool


def should_abstain(
    retrieval_confidence: float,
    verification: VerificationSummary,
    retrieval_threshold: float = 0.40,
) -> AbstentionDecision:
    """Apply bounded abstention policy.

    Reasons are explicit so downstream logs and dashboards can group failure modes.
    """
    if retrieval_confidence < retrieval_threshold:
        return AbstentionDecision(True, "low_retrieval_confidence")

    if not verification.is_grounded:
        return AbstentionDecision(True, "not_grounded")

    if verification.conflict_detected:
        return AbstentionDecision(True, "conflicting_sources")

    return AbstentionDecision(False, None)
