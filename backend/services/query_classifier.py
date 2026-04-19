"""Query classification contracts for routing-aware retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


QueryType = Literal["factual", "ambiguous", "multi_hop", "exception"]


@dataclass(frozen=True)
class QueryClassification:
    type: QueryType
    confidence: float


class QueryClassifier(Protocol):
    def classify(self, query: str) -> QueryClassification:
        ...


class HeuristicQueryClassifier:
    """Simple baseline classifier.

    This is intentionally conservative and fully deterministic.
    """

    def classify(self, query: str) -> QueryClassification:
        q = (query or "").lower().strip()
        if not q:
            return QueryClassification(type="ambiguous", confidence=0.5)

        if any(k in q for k in ["if", "exception", "unless", "special case"]):
            return QueryClassification(type="exception", confidence=0.7)

        if any(k in q for k in ["and", "also", "compare", "between", "versus"]):
            return QueryClassification(type="multi_hop", confidence=0.65)

        if len(q.split()) <= 6:
            return QueryClassification(type="factual", confidence=0.7)

        return QueryClassification(type="ambiguous", confidence=0.6)
