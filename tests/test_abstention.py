from backend.services.abstention import VerificationSummary, should_abstain


def test_abstain_for_low_retrieval_confidence() -> None:
    decision = should_abstain(
        retrieval_confidence=0.2,
        verification=VerificationSummary(is_grounded=True, conflict_detected=False),
    )
    assert decision.abstained is True
    assert decision.reason == "low_retrieval_confidence"


def test_abstain_for_not_grounded() -> None:
    decision = should_abstain(
        retrieval_confidence=0.9,
        verification=VerificationSummary(is_grounded=False, conflict_detected=False),
    )
    assert decision.abstained is True
    assert decision.reason == "not_grounded"


def test_abstain_for_conflict() -> None:
    decision = should_abstain(
        retrieval_confidence=0.9,
        verification=VerificationSummary(is_grounded=True, conflict_detected=True),
    )
    assert decision.abstained is True
    assert decision.reason == "conflicting_sources"


def test_no_abstain_when_checks_pass() -> None:
    decision = should_abstain(
        retrieval_confidence=0.9,
        verification=VerificationSummary(is_grounded=True, conflict_detected=False),
    )
    assert decision.abstained is False
    assert decision.reason is None
