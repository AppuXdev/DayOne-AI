from eval import QueryResult, _summarise


def _make_result(query_id: str, expect_answer: bool, answer: str) -> QueryResult:
    return QueryResult(
        id=query_id,
        category="test",
        query="q",
        expect_answer=expect_answer,
        mode="reranker_on",
        answer=answer,
        sources_cited=1,
        confidence=0.5,
        latency_ms=10.0,
        ttft_ms=5.0,
        retrieval_hit=expect_answer and "i do not have" not in answer.lower(),
        correct_abstain=(not expect_answer) and ("i do not have" in answer.lower()),
        error_category="ok",
    )


def test_abstention_precision_recall_computation() -> None:
    # TP: negative abstained
    tp = _make_result("n1", False, "I do not have that information in the current HR files. Please contact HR.")
    # FN: negative answered
    fn = _make_result("n2", False, "We provide relocation bonuses.")
    # FP: positive abstained (false abstention)
    fp = _make_result("p1", True, "I do not have that information in the current HR files. Please contact HR.")
    # TN (for abstain class): positive answered
    tn = _make_result("p2", True, "Employees get 20 PTO days annually.")

    summary = _summarise([tp, fn, fp, tn], mode="reranker_on")

    assert summary["abstention_precision"] == 0.5
    assert summary["abstention_recall"] == 0.5
    assert summary["abstention_f1"] == 0.5
    assert summary["false_abstentions"] == 1
    assert summary["false_abstention_rate"] == 0.5
