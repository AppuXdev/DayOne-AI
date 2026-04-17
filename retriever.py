"""Hybrid retrieval for DayOne AI.

Architecture:
  1. Dense retrieval  — FAISS (all-MiniLM-L6-v2 embeddings)
  2. Sparse retrieval — BM25Okapi on same corpus
  3. Fusion           — Reciprocal Rank Fusion (RRF, k=60)
  4. Reranking        — cross-encoder/ms-marco-MiniLM-L-6-v2 (toggleable)
  5. Confidence       — sigmoid(top_reranker_score) when reranker ON;
                        normalised BM25 score when reranker OFF

Design trade-offs documented inline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config — override via environment variable
# ---------------------------------------------------------------------------
# Trade-off: reranker adds ~200-400 ms CPU latency but improves precision by
# ~10-16 pp on our benchmark (see eval.py results). Default ON for correctness.
USE_RERANKER: bool = os.getenv("DAYONE_USE_RERANKER", "1") != "0"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Retrieve this many candidates before reranking. Wider net = better recall
# at the cost of reranker compute. 12 balances recall vs latency on CPU.
CANDIDATE_K: int = 12
FINAL_K: int = 4       # Docs passed to the LLM
RRF_K: int = 60        # RRF smoothing constant (standard value from literature)

# Confidence thresholds
CONF_LOW: float = 0.40   # Below this → show "low confidence" warning in UI
CONF_HIGH: float = 0.70  # Above this → high confidence

_cross_encoder_cache: Dict[str, Any] = {}


@dataclass
class RetrievalResult:
    """Structured result from HybridRetriever.retrieve().

    Carries everything needed for the Answer Justification layer:
    - final_docs / final_scores  : what the LLM sees
    - candidates / candidate_scores : what existed before reranking
    - rank_changes               : index delta per final doc
    """
    final_docs: List[Any]          # Top final_k docs sent to LLM
    final_scores: List[float]      # CE scores (or BM25 proxies) for final docs
    confidence: float              # [0, 1] confidence estimate
    candidates: List[Any]          # All candidate_k docs before reranking
    candidate_scores: List[float]  # Scores of candidates (BM25 or CE pre-sort)
    latency_ms: float
    used_reranker: bool

    @property
    def rank_changes(self) -> List[int]:
        """For each final doc, how many positions it moved up during reranking.

        Positive = moved up (promoted by reranker). Zero = same position.
        Allows UI to show ↑N next to each chunk.
        """
        changes: List[int] = []
        for doc in self.final_docs:
            try:
                candidate_pos = next(
                    i for i, c in enumerate(self.candidates)
                    if c.page_content == doc.page_content
                )
                final_pos = self.final_docs.index(doc)
                changes.append(candidate_pos - final_pos)
            except StopIteration:
                changes.append(0)
        return changes


def _get_cross_encoder() -> Any:
    """Lazy-load and cache the cross-encoder (one instance per process)."""
    if RERANKER_MODEL not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        _cross_encoder_cache[RERANKER_MODEL] = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder_cache[RERANKER_MODEL]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class HybridRetriever:
    """BM25 + FAISS hybrid retriever with optional cross-encoder reranking.

    Why hybrid instead of pure dense?
    Dense retrieval (FAISS) struggles with exact keyword matches such as
    policy form numbers, HR codes, and proper nouns. BM25 handles these
    well. RRF fusion gets the best of both without requiring score
    normalisation across incompatible scales.
    """

    def __init__(
        self,
        vector_store: FAISS,
        embeddings: Any,
        use_reranker: bool = USE_RERANKER,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.use_reranker = use_reranker
        # Optional per-source reputation weights from FeedbackStore.
        # Keys are bare filenames (e.g. "handbook.pdf"); values centred at 1.0.
        self._source_weights: Dict[str, float] = source_weights or {}

        # Build ordered corpus aligned with FAISS internal indices.
        # FAISS internal index i → index_to_docstore_id[i] → docstore doc.
        n = vector_store.index.ntotal
        self._docs: List[Any] = []
        for i in range(n):
            doc_id = vector_store.index_to_docstore_id[i]
            self._docs.append(vector_store.docstore._dict[doc_id])

        # BM25 on same corpus order — indices are directly comparable.
        # Why BM25Okapi? It applies IDF and term-frequency saturation,
        # outperforming plain TF-IDF for short query / long document settings.
        corpus = [doc.page_content.lower().split() for doc in self._docs]
        self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_indices(self, query: str, k: int) -> List[int]:
        """Return FAISS internal indices for the top-k dense results."""
        query_vec = np.array(
            [self.embeddings.embed_query(query)], dtype=np.float32
        )
        _, faiss_indices = self.vector_store.index.search(query_vec, k)
        return [int(i) for i in faiss_indices[0] if i >= 0]

    def _sparse_indices(self, query: str, k: int) -> Tuple[List[int], "np.ndarray"]:
        """Return BM25-ranked indices and the full score array."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indices = list(np.argsort(scores)[::-1][:k])
        return indices, scores

    def _apply_source_weights(self, docs: List[Any], scores: List[float]) -> List[float]:
        """Multiply each score by its source's reputation weight.

        Looks up each doc's source filename in self._source_weights.
        Unknown sources get weight 1.0 (neutral).
        """
        if not self._source_weights:
            return scores
        from pathlib import Path as _Path
        weighted: List[float] = []
        for doc, score in zip(docs, scores):
            source_name = _Path(str(doc.metadata.get("source", ""))).name
            weight = self._source_weights.get(source_name, 1.0)
            weighted.append(score * weight)
        return weighted

    @staticmethod
    def _rrf_fuse(
        dense: List[int], sparse: List[int], k: int = RRF_K
    ) -> List[int]:
        """Reciprocal Rank Fusion across two ranked lists.

        score(d) = Σ  1 / (rank(d, list) + k)
        k=60 is the standard value from Cormack et al. (2009).
        """
        scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
        for rank, idx in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_k: int = FINAL_K,
        candidate_k: int = CANDIDATE_K,
    ) -> RetrievalResult:
        """Run hybrid retrieval and optional reranking.

        Returns a RetrievalResult with both pre- and post-reranking data,
        enabling the Answer Justification layer in the UI.
        """
        t0 = time.perf_counter()

        dense_idx = self._dense_indices(query, candidate_k)
        sparse_idx, bm25_scores = self._sparse_indices(query, candidate_k)
        fused = self._rrf_fuse(dense_idx, sparse_idx)[:candidate_k]
        candidates = [self._docs[i] for i in fused if i < len(self._docs)]
        cand_bm25 = [float(bm25_scores[i]) for i in fused if i < len(self._docs)]

        if not candidates:
            return RetrievalResult(
                final_docs=[], final_scores=[], confidence=0.0,
                candidates=[], candidate_scores=[], latency_ms=0.0,
                used_reranker=self.use_reranker,
            )

        if self.use_reranker:
            ce = _get_cross_encoder()
            pairs = [(query, doc.page_content) for doc in candidates]
            raw_scores: List[float] = ce.predict(pairs).tolist()
            # Apply source-reputation weights before final ranking
            weighted_scores = self._apply_source_weights(candidates, raw_scores)
            ranked = sorted(
                zip(candidates, weighted_scores, raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            final_docs = [d for d, _, _ in ranked[:final_k]]
            final_scores = [ws for _, ws, _ in ranked[:final_k]]  # weighted scores shown in UI
            confidence = _sigmoid(raw_scores[candidates.index(final_docs[0])]) if final_docs else 0.0
        else:
            weighted_bm25 = self._apply_source_weights(candidates, cand_bm25)
            ranked_pairs = sorted(
                zip(candidates, weighted_bm25, cand_bm25),
                key=lambda x: x[1],
                reverse=True,
            )
            final_docs = [d for d, _, _ in ranked_pairs[:final_k]]
            final_scores = [ws for _, ws, _ in ranked_pairs[:final_k]]
            top_bm25 = cand_bm25[0] if cand_bm25 else 0.0
            confidence = min(0.30 + (top_bm25 / 15.0) * 0.60, 0.95)
            raw_scores = cand_bm25  # for candidate_scores below

        latency_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            final_docs=final_docs,
            final_scores=final_scores,
            confidence=confidence,
            candidates=candidates,
            candidate_scores=raw_scores if self.use_reranker else cand_bm25,
            latency_ms=latency_ms,
            used_reranker=self.use_reranker,
        )


def confidence_label(score: float) -> str:
    """Human-readable confidence label for UI display."""
    if score >= CONF_HIGH:
        return "high"
    if score >= CONF_LOW:
        return "medium"
    return "low"
