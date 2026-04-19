from backend.services.query_classifier import HeuristicQueryClassifier
from backend.services.query_router import route_query


def test_classifier_flags_factual_query() -> None:
    classifier = HeuristicQueryClassifier()
    result = classifier.classify("What is PTO?")
    assert result.type == "factual"
    assert 0.0 <= result.confidence <= 1.0


def test_classifier_flags_multi_hop_query() -> None:
    classifier = HeuristicQueryClassifier()
    result = classifier.classify("Compare leave and benefits policies")
    assert result.type == "multi_hop"


def test_router_expands_ambiguous_queries() -> None:
    config = route_query("ambiguous")
    assert config.name == "expanded"
    assert config.candidate_k == 20


def test_router_uses_fast_path_for_factual_queries() -> None:
    config = route_query("factual")
    assert config.name == "fast_path"
    assert config.candidate_k == 8
