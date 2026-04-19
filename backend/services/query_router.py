"""Route selection for query-adaptive retrieval configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.services.query_classifier import QueryType


RouteName = Literal["fast_path", "expanded", "iterative"]


@dataclass(frozen=True)
class RouteConfig:
    name: RouteName
    candidate_k: int
    iterative_steps: int


def route_query(query_type: QueryType) -> RouteConfig:
    """Map query class to bounded retrieval strategy."""
    if query_type == "factual":
        return RouteConfig(name="fast_path", candidate_k=8, iterative_steps=1)

    if query_type == "ambiguous":
        return RouteConfig(name="expanded", candidate_k=20, iterative_steps=1)

    if query_type in {"multi_hop", "exception"}:
        return RouteConfig(name="iterative", candidate_k=16, iterative_steps=2)

    return RouteConfig(name="expanded", candidate_k=12, iterative_steps=1)
