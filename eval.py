"""DayOne AI evaluation harness — two-tier ML evaluation.

Tier 1 — Default (fast, deterministic, always runs)
----------------------------------------------------
  - Retrieval hit rate
  - Correct abstentions (negative queries)
  - Precision@1 / @3 / @k  (proxy: keyword-in-chunk)
  - Avg latency (total) and TTFT (time-to-first-token)
  - Avg confidence
  - Error category breakdown

Tier 2 — Judge Mode (opt-in via --judge)
-----------------------------------------
  - Faithfulness score  (claim-by-claim context grounding via LLM)
  - Answer correctness  (does the answer address the question?)
  - Hallucination flag  (any claim not supported by context?)
  - Results cached to eval_judge_cache.json — only uncached queries
    are re-sent to the LLM, keeping repeat runs cheap

Usage
-----
    python eval.py --org org_acme
    python eval.py --org org_acme --output eval_results.json
    python eval.py --org org_acme --judge          # + Tier 2 metrics
    python eval.py --org org_acme --judge --rerun  # ignore cache

Design notes
------------
- Benchmark queries are HARD-CODED (not LLM-generated) to avoid
  distribution leakage and ensure reproducibility.
- Faithfulness evaluation is opt-in because it requires one extra Groq
  call per query, introduces mild non-determinism, and slows the eval
  loop. Baseline metrics remain fully reproducible.
- TTFT is measured as the wall-clock time from query dispatch to receipt
  of the first token. This is the latency users perceive, not the total
  generation time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3-8b-8192"
JUDGE_CACHE_PATH = ROOT_DIR / "eval_judge_cache.json"

FALLBACK_PHRASES = [
    "i do not have that information",
    "not in the current hr files",
    "knowledge base is currently empty",
    "please contact hr",
]

SYSTEM_PROMPT = (
    "You are DayOne AI, a professional HR onboarding assistant. "
    "Answer ONLY from the retrieved context below. "
    "If the answer is not in the context, say exactly: "
    "'I do not have that information in the current HR files. Please contact HR.' "
    "Do not invent or infer beyond what is explicitly stated."
)


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    id: str
    category: str          # direct | paraphrase | multi_hop | negative | ambiguous
    query: str
    keywords: List[str]    # Expected topic keywords (for hit detection)
    expect_answer: bool    # True = system should find info; False = should abstain


BENCHMARK: List[BenchmarkQuery] = [
    # ── Direct retrieval ──────────────────────────────────────────────────
    BenchmarkQuery("Q01", "direct",
                   "How many PTO days do employees get per year?",
                   ["pto", "vacation", "days", "leave", "paid"], True),
    BenchmarkQuery("Q02", "direct",
                   "What is the process to request sick leave?",
                   ["sick", "leave", "request", "notify", "process"], True),
    BenchmarkQuery("Q03", "direct",
                   "What health insurance options are available to employees?",
                   ["health", "insurance", "medical", "coverage", "plan"], True),
    BenchmarkQuery("Q04", "direct",
                   "What are the standard working hours?",
                   ["hours", "schedule", "working", "core", "office"], True),
    BenchmarkQuery("Q05", "direct",
                   "What is the probationary period for new hires?",
                   ["probation", "period", "new", "hire", "90", "days"], True),

    # ── Paraphrased queries ───────────────────────────────────────────────
    BenchmarkQuery("Q06", "paraphrase",
                   "How much paid time off am I entitled to annually?",
                   ["pto", "vacation", "days", "leave", "annual"], True),
    BenchmarkQuery("Q07", "paraphrase",
                   "Can you explain my medical coverage options?",
                   ["health", "insurance", "medical", "coverage", "plan"], True),
    BenchmarkQuery("Q08", "paraphrase",
                   "What time do I need to be in the office each day?",
                   ["hours", "schedule", "office", "core", "start"], True),
    BenchmarkQuery("Q09", "paraphrase",
                   "When am I eligible to take leave after I join?",
                   ["probation", "leave", "eligible", "join", "after"], True),

    # ── Multi-hop questions ───────────────────────────────────────────────
    BenchmarkQuery("Q10", "multi_hop",
                   "What happens to unused PTO at the end of the year?",
                   ["pto", "unused", "carry", "rollover", "expire", "year"], True),
    BenchmarkQuery("Q11", "multi_hop",
                   "If I fall sick during an approved vacation, can I reclaim those days?",
                   ["sick", "vacation", "reclaim", "overlap", "days"], True),
    BenchmarkQuery("Q12", "multi_hop",
                   "What is the process to add a dependent to my health plan after joining?",
                   ["dependent", "health", "plan", "add", "enrollment"], True),

    # ── Negative queries (not expected in typical HR docs) ────────────────
    BenchmarkQuery("Q13", "negative",
                   "Does the company provide a relocation bonus?",
                   ["relocation", "bonus", "moving", "allowance"], False),
    BenchmarkQuery("Q14", "negative",
                   "What is the company's policy on cryptocurrency salary payments?",
                   ["crypto", "bitcoin", "cryptocurrency", "salary"], False),
    BenchmarkQuery("Q15", "negative",
                   "Can I permanently work from a different country?",
                   ["country", "international", "permanent", "remote", "abroad"], False),
    BenchmarkQuery("Q16", "negative",
                   "What is the severance package for employees who are laid off?",
                   ["severance", "layoff", "termination", "package", "redundancy"], False),

    # ── Ambiguous queries (tests retrieval quality on broad topics) ───────
    BenchmarkQuery("Q17", "ambiguous",
                   "What are the benefits?",
                   ["benefits", "health", "leave", "insurance", "perks"], True),
    BenchmarkQuery("Q18", "ambiguous",
                   "How does leave work here?",
                   ["leave", "pto", "vacation", "sick", "days"], True),
    BenchmarkQuery("Q19", "ambiguous",
                   "Tell me about the onboarding process.",
                   ["onboarding", "new hire", "first day", "orientation", "start"], True),
    BenchmarkQuery("Q20", "ambiguous",
                   "What should I know about insurance?",
                   ["insurance", "health", "dental", "vision", "coverage", "plan"], True),
]


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = [
    "no_retrieval",       # Retriever returned 0 docs
    "retrieval_miss",     # Docs returned but no relevant keyword found
    "abstention_error",   # System said IDK when it should have answered
    "hallucination",      # Judge flagged unsupported claim (--judge only)
    "ok",                 # Correct
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    id: str
    category: str
    query: str
    expect_answer: bool
    mode: str               # "reranker_on" | "reranker_off"
    answer: str
    sources_cited: int
    confidence: float
    latency_ms: float
    ttft_ms: float          # Time-to-first-token (streaming mode)
    retrieval_hit: bool
    correct_abstain: bool
    error_category: str     # One of ERROR_CATEGORIES
    keywords_found: List[str] = field(default_factory=list)
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_k: float = 0.0
    # Tier 2 — populated only with --judge
    faithfulness: Optional[float] = None
    correctness: Optional[float] = None
    hallucination: Optional[bool] = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _is_fallback(answer: str) -> bool:
    low = answer.lower()
    return any(phrase in low for phrase in FALLBACK_PHRASES)


def _keywords_in_answer(answer: str, keywords: List[str]) -> List[str]:
    low = answer.lower()
    return [kw for kw in keywords if kw in low]


def _classify_error(
    bq: BenchmarkQuery,
    sources_cited: int,
    answer: str,
    keywords_found: List[str],
) -> str:
    fell_back = _is_fallback(answer)
    if bq.expect_answer:
        if sources_cited == 0:
            return "no_retrieval"
        if fell_back or not keywords_found:
            return "retrieval_miss"
        return "ok"
    else:
        # Negative query — correct behaviour is to abstain
        if not fell_back:
            return "abstention_error"
        return "ok"


def _build_retriever(org_id: str, use_reranker: bool):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from retriever import HybridRetriever

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    store_path = VECTOR_STORE_DIR / org_id
    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found for org '{org_id}'. Run python ingest.py first."
        )
    store = FAISS.load_local(
        str(store_path), embeddings, allow_dangerous_deserialization=True
    )
    return HybridRetriever(store, embeddings, use_reranker=use_reranker), embeddings


def _build_llm():
    from langchain_groq import ChatGroq
    from pydantic import SecretStr

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to .env.")
    return ChatGroq(model=MODEL_NAME, temperature=0, api_key=SecretStr(api_key))


# ---------------------------------------------------------------------------
# Tier 1 — retrieval + generation
# ---------------------------------------------------------------------------

def run_query(
    bq: BenchmarkQuery,
    retriever,
    llm,
    mode: str,
) -> QueryResult:
    """Run a single benchmark query and return a structured Tier 1 result."""
    from langchain.schema import HumanMessage, SystemMessage
    from retriever import RetrievalResult

    result: RetrievalResult = retriever.retrieve(bq.query)
    docs = result.final_docs
    confidence = result.confidence
    latency_ms = result.latency_ms

    # Precision@k proxy — keyword in chunk text approximates relevance
    p_at_k: List[float] = []
    for k in (1, 3, len(docs)):
        if k == 0:
            p_at_k.append(0.0)
            continue
        top = docs[:k]
        hits = sum(
            any(kw in d.page_content.lower() for kw in bq.keywords)
            for d in top
        )
        p_at_k.append(hits / k if top else 0.0)

    ttft_ms = 0.0

    if not docs:
        answer = "I do not have that information in the current HR files. Please contact HR."
        sources_cited = 0
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
            for d in docs
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {bq.query}\n\nContext:\n{context}"),
        ]

        # Measure TTFT via streaming
        t_dispatch = time.perf_counter()
        first_token_received = False
        answer_parts: List[str] = []
        try:
            for chunk in llm.stream(messages):
                if not first_token_received:
                    ttft_ms = (time.perf_counter() - t_dispatch) * 1000
                    first_token_received = True
                answer_parts.append(chunk.content)
        except Exception:
            # Fall back to non-streaming if stream fails
            t0 = time.perf_counter()
            response = llm.invoke(messages)
            ttft_ms = (time.perf_counter() - t0) * 1000
            answer_parts = [response.content]

        answer = "".join(answer_parts).strip()
        latency_ms += (time.perf_counter() - t_dispatch) * 1000
        sources_cited = len(docs)

    fell_back = _is_fallback(answer)
    retrieval_hit = (sources_cited > 0) and (not fell_back) if bq.expect_answer else False
    correct_abstain = fell_back if not bq.expect_answer else False
    keywords_found = _keywords_in_answer(answer, bq.keywords)
    error_category = _classify_error(bq, sources_cited, answer, keywords_found)

    return QueryResult(
        id=bq.id,
        category=bq.category,
        query=bq.query,
        expect_answer=bq.expect_answer,
        mode=mode,
        answer=answer,
        sources_cited=sources_cited,
        confidence=confidence,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        retrieval_hit=retrieval_hit,
        correct_abstain=correct_abstain,
        error_category=error_category,
        keywords_found=keywords_found,
        precision_at_1=p_at_k[0] if p_at_k else 0.0,
        precision_at_3=p_at_k[1] if len(p_at_k) > 1 else 0.0,
        precision_at_k=p_at_k[2] if len(p_at_k) > 2 else 0.0,
    )


# ---------------------------------------------------------------------------
# Tier 2 — LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_FAITHFULNESS_PROMPT = """\
You are an evaluation judge for a RAG system.

QUESTION: {question}

CONTEXT (retrieved documents):
{context}

ANSWER: {answer}

Task:
1. Decompose the ANSWER into individual factual claims.
2. For each claim, determine if it is directly supported by the CONTEXT.
3. Return a JSON object with:
   - "faithfulness": float 0.0–1.0 (fraction of claims supported by context)
   - "correctness": float 0.0–1.0 (does the answer actually address the question?)
   - "hallucination": boolean (true if ANY claim is NOT supported by context)
   - "reasoning": string (brief one-line explanation)

Respond ONLY with valid JSON. No markdown fences."""

JUDGE_ABSTAIN_PROMPT = """\
You are an evaluation judge. The system was asked a question that has NO answer in the knowledge base.

QUESTION: {question}
SYSTEM RESPONSE: {answer}

Task: Did the system correctly decline to answer?
Return JSON: {{"correct_abstention": true/false, "faithfulness": 1.0 or 0.0, "correctness": 1.0 or 0.0, "hallucination": false, "reasoning": "..."}}
Respond ONLY with valid JSON."""


def _load_judge_cache() -> dict:
    if JUDGE_CACHE_PATH.exists():
        try:
            return json.loads(JUDGE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_judge_cache(cache: dict) -> None:
    JUDGE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(query_id: str, mode: str) -> str:
    return f"{query_id}::{mode}"


def run_judge(
    result: QueryResult,
    docs: list,
    llm,
    cache: dict,
    rerun: bool,
) -> QueryResult:
    """Run Tier 2 LLM-as-judge evaluation and attach scores to the result.

    Results are cached by (query_id, mode) to avoid re-paying LLM cost
    on repeat eval runs. Pass rerun=True to bypass the cache.
    """
    from langchain.schema import HumanMessage

    key = _cache_key(result.id, result.mode)
    if not rerun and key in cache:
        cached = cache[key]
        result.faithfulness = cached.get("faithfulness")
        result.correctness = cached.get("correctness")
        result.hallucination = cached.get("hallucination")
        return result

    context = "\n\n---\n\n".join(
        f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in docs
    ) if docs else "(no context retrieved)"

    if result.expect_answer:
        prompt = JUDGE_FAITHFULNESS_PROMPT.format(
            question=result.query,
            context=context[:3000],
            answer=result.answer[:1000],
        )
    else:
        prompt = JUDGE_ABSTAIN_PROMPT.format(
            question=result.query,
            answer=result.answer[:1000],
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown fences if the model includes them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        result.faithfulness = float(scores.get("faithfulness", 0.0))
        result.correctness = float(scores.get("correctness", 0.0))
        result.hallucination = bool(scores.get("hallucination", False))
        if result.hallucination:
            result.error_category = "hallucination"
        cache[key] = {
            "faithfulness": result.faithfulness,
            "correctness": result.correctness,
            "hallucination": result.hallucination,
        }
    except Exception as exc:
        print(f"    [judge] failed for {result.id}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summarise(results: List[QueryResult], mode: str) -> dict:
    positives = [r for r in results if r.expect_answer]
    negatives = [r for r in results if not r.expect_answer]

    hit_rate = (
        sum(r.retrieval_hit for r in positives) / len(positives)
        if positives else 0.0
    )
    abstain_rate = (
        sum(r.correct_abstain for r in negatives) / len(negatives)
        if negatives else 0.0
    )
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0
    avg_ttft = sum(r.ttft_ms for r in results) / len(results) if results else 0.0
    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0.0
    avg_p1 = sum(r.precision_at_1 for r in positives) / len(positives) if positives else 0.0
    avg_p3 = sum(r.precision_at_3 for r in positives) / len(positives) if positives else 0.0
    avg_pk = sum(r.precision_at_k for r in positives) / len(positives) if positives else 0.0

    error_counts = {cat: 0 for cat in ERROR_CATEGORIES}
    for r in results:
        cat = r.error_category if r.error_category in error_counts else "ok"
        error_counts[cat] += 1

    # Tier 2 metrics (None if --judge not used)
    judged = [r for r in results if r.faithfulness is not None]
    avg_faith = sum(r.faithfulness for r in judged) / len(judged) if judged else None  # type: ignore[arg-type]
    avg_corr = sum(r.correctness for r in judged) / len(judged) if judged else None    # type: ignore[arg-type]
    halluc_rate = sum(1 for r in judged if r.hallucination) / len(judged) if judged else None

    return {
        "mode": mode,
        "n_queries": len(results),
        # Tier 1
        "positive_hit_rate": round(hit_rate, 3),
        "negative_abstain_rate": round(abstain_rate, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "avg_confidence": round(avg_conf, 3),
        "precision_at_1": round(avg_p1, 3),
        "precision_at_3": round(avg_p3, 3),
        "precision_at_k": round(avg_pk, 3),
        "error_categories": error_counts,
        # Tier 2
        "avg_faithfulness": round(avg_faith, 3) if avg_faith is not None else None,
        "avg_correctness": round(avg_corr, 3) if avg_corr is not None else None,
        "hallucination_rate": round(halluc_rate, 3) if halluc_rate is not None else None,
    }


def _print_table(summary_on: dict, summary_off: dict) -> None:
    W = 26
    print("\n" + "=" * 76)
    print("  DayOne AI — Evaluation Results")
    print("=" * 76)
    print(f"{'Metric':<{W}} {'Reranker ON':>22} {'Reranker OFF':>22}")
    print("-" * 76)

    tier1_metrics = [
        ("Retrieval hit rate",     "positive_hit_rate",     "{:.1%}"),
        ("Correct abstentions",    "negative_abstain_rate",  "{:.1%}"),
        ("Precision@1 (proxy)",    "precision_at_1",         "{:.1%}"),
        ("Precision@3 (proxy)",    "precision_at_3",         "{:.1%}"),
        ("Precision@k (proxy)",    "precision_at_k",         "{:.1%}"),
        ("Avg latency (ms)",       "avg_latency_ms",         "{:.0f} ms"),
        ("Avg TTFT (ms)",          "avg_ttft_ms",            "{:.0f} ms"),
        ("Avg confidence",         "avg_confidence",         "{:.3f}"),
        ("Queries run",            "n_queries",              "{}"),
    ]
    for label, key, fmt in tier1_metrics:
        val_on = fmt.format(summary_on[key])
        val_off = fmt.format(summary_off[key])
        print(f"{label:<{W}} {val_on:>22} {val_off:>22}")

    # Tier 2 — only printed if judge was run
    if summary_on.get("avg_faithfulness") is not None:
        print("-" * 76)
        print(f"{'— Tier 2: Judge Metrics —':<{W}}")
        tier2_metrics = [
            ("Faithfulness",           "avg_faithfulness",       "{:.1%}"),
            ("Answer correctness",     "avg_correctness",        "{:.1%}"),
            ("Hallucination rate",     "hallucination_rate",     "{:.1%}"),
        ]
        for label, key, fmt in tier2_metrics:
            v_on = summary_on.get(key)
            v_off = summary_off.get(key)
            val_on = fmt.format(v_on) if v_on is not None else "—"
            val_off = fmt.format(v_off) if v_off is not None else "—"
            print(f"{label:<{W}} {val_on:>22} {val_off:>22}")

    print("=" * 76)

    # Error category breakdown
    print("\nError category breakdown (Reranker ON):")
    cats = summary_on["error_categories"]
    for cat, count in cats.items():
        bar = "█" * count
        print(f"  {cat:<20} {count:>3}  {bar}")

    print("\nNotes:")
    print("  - Precision@k (proxy): proportion of top-k chunks containing at")
    print("    least one expected keyword. NOT a labelled-corpus metric.")
    print("  - TTFT: time-to-first-token (perceived latency).")
    if summary_on.get("avg_faithfulness") is not None:
        print("  - Faithfulness / correctness: LLM-as-judge scores (Groq). Non-")
        print("    deterministic; use --judge sparingly and trust the Tier 1 baseline.")
    else:
        print("  - Faithfulness evaluation available via: python eval.py --judge")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DayOne AI evaluation harness")
    parser.add_argument("--org", required=True, help="Organisation ID (e.g. org_acme)")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "eval_results.json"),
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable Tier 2 LLM-as-judge faithfulness/correctness scoring (opt-in, cached)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore judge cache and re-evaluate all queries (use with --judge)",
    )
    args = parser.parse_args()

    llm = _build_llm()
    judge_cache = _load_judge_cache() if args.judge else {}
    all_results: List[dict] = []
    summaries: List[dict] = []

    for use_reranker, mode_label in [(True, "reranker_on"), (False, "reranker_off")]:
        print(f"\n[eval] Running mode: {mode_label} ...")
        retriever, _ = _build_retriever(args.org, use_reranker=use_reranker)
        mode_results: List[QueryResult] = []

        for bq in BENCHMARK:
            print(f"  {bq.id} ({bq.category}): {bq.query[:55]}...")
            result = run_query(bq, retriever, llm, mode_label)

            if args.judge:
                # Re-retrieve docs for judge context
                from retriever import RetrievalResult
                rr: RetrievalResult = retriever.retrieve(bq.query)
                result = run_judge(result, rr.final_docs, llm, judge_cache, args.rerun)

            mode_results.append(result)
            status = "✓" if result.error_category == "ok" else f"✗ [{result.error_category}]"
            faith_str = f"  faith={result.faithfulness:.2f}" if result.faithfulness is not None else ""
            print(
                f"    {status} conf={result.confidence:.2f}  "
                f"lat={result.latency_ms:.0f}ms  ttft={result.ttft_ms:.0f}ms  "
                f"sources={result.sources_cited}{faith_str}"
            )

        summary = _summarise(mode_results, mode_label)
        summaries.append(summary)
        all_results.extend([asdict(r) for r in mode_results])

    if args.judge:
        _save_judge_cache(judge_cache)
        print(f"[eval] Judge cache saved → {JUDGE_CACHE_PATH}")

    _print_table(summaries[0], summaries[1])

    output = {
        "benchmark_size": len(BENCHMARK),
        "categories": ["direct", "paraphrase", "multi_hop", "negative", "ambiguous"],
        "judge_enabled": args.judge,
        "summaries": summaries,
        "results": all_results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[eval] Results written to {args.output}")


if __name__ == "__main__":
    main()
