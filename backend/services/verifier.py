"""Verification contracts for grounded-answer checks.

This is a service-layer contract; concrete model/prompt execution can be wired later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Sequence


@dataclass(frozen=True)
class VerificationResult:
    is_grounded: bool
    verification_confidence: float
    unsupported_claims: List[str] = field(default_factory=list)
    conflict_detected: bool = False


class AnswerVerifier(Protocol):
    """Interface for verification model adapters."""

    def verify(self, query: str, answer: str, retrieved_chunks: Sequence[str]) -> VerificationResult:
        ...


class NullVerifier:
    """Safe fallback verifier used before model integration.

    It marks answers as grounded only when non-empty context exists.
    """

    def verify(self, query: str, answer: str, retrieved_chunks: Sequence[str]) -> VerificationResult:
        has_context = any((c or "").strip() for c in retrieved_chunks)
        if not has_context:
            return VerificationResult(
                is_grounded=False,
                verification_confidence=0.0,
                unsupported_claims=["No retrieved context provided to verifier."],
                conflict_detected=False,
            )

        return VerificationResult(
            is_grounded=True,
            verification_confidence=0.5,
            unsupported_claims=[],
            conflict_detected=False,
        )
