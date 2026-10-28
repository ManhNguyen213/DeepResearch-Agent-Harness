"""Hybrid retrieval core: Qdrant dense + BM25 sparse + Reciprocal Rank Fusion + optional reranker.

Data Flow:
    query -> [Qdrant dense top-N] + [BM25 top-N] -> date filter -> Reciprocal Rank Fusion
          -> (optional Cross-Encoder rerank) -> top-k ``RetrievedChunk`` with deterministic ``chunk_id``.

Design Rationale:
    GR1's retrieval stack is re-implemented here as a transparent, dependency-light module so the
    MCP server stays thin. Dense and sparse rankers are fused with RRF (rank-based, so scores from
    different scales need no calibration). Every chunk carries a stable ``chunk_id`` which the
    Verification Gate later checks deterministically against ``get_chunk_by_id``.
"""

import json
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rank_bm25 import BM25Okapi

from src.memory.embeddings import Embedder
from src.schemas.evidence import RetrievedChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
RRF_K = 60


class Reranker(Protocol):
    """Reranker interface: score each candidate chunk against the query."""

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Return one relevance score per chunk (higher is better)."""
        ...


class CrossEncoderReranker:
    """Cross-Encoder reranker (requires the optional ``sentence-transformers`` package)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Score (query, chunk) pairs with the cross-encoder."""
        pairs = [(query, c.content) for c in chunks]
        return [float(s) for s in self._model.predict(pairs)]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_chunks_from_dir(corpus_dir: Path | str) -> list[RetrievedChunk]:
    """Load chunks from every ``*.jsonl`` file in a directory.

    Each line must contain ``chunk_id``, ``content`` and optionally ``document_title`` and
    ``published_date``.
    """
    chunks: list[RetrievedChunk] = []
    for file in sorted(Path(corpus_dir).glob("*.jsonl")):
        with file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(RetrievedChunk(**json.loads(line)))
    return chunks


def date_matches(published_date: str | None, date_filter: str | None) -> bool:
    """Check ``published_date`` against a filter.

    Supported filters: a prefix (``"2025"``, ``"2025-06"``, ``"2025-06-01"``) or an inclusive
    range ``"start..end"`` where either bound may be empty (e.g. ``"2024-01-01.."``).
    Chunks without a date never match a non-empty filter.
    """
    if not date_filter:
        return True
    if not published_date:
        return False
    if ".." in date_filter:
        start, end = date_filter.split("..", 1)
        if start and published_date < start:
            return False
        return not (end and published_date > end)
    return published_date.startswith(date_filter)


class HybridRetriever:
    """In-process Hybrid RAG retriever over a fixed chunk corpus."""

    COLLECTION = "document_chunks"

    def __init__(
        self,
        chunks: Iterable[RetrievedChunk],
        qdrant_client: QdrantClient,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self._chunks: dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
        self._ids: list[str] = list(self._chunks)
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = BM25Okapi([_tokenize(self._chunks[i].content) for i in self._ids]) if self._ids else None
        self._index_dense()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _index_dense(self) -> None:
        """(Re)build the dense index unless a persisted collection already matches the corpus."""
        if self._qdrant.collection_exists(self.COLLECTION):
            info = self._qdrant.get_collection(self.COLLECTION)
            vectors = info.config.params.vectors
            same_dim = getattr(vectors, "size", None) == self._embedder.dim
            if same_dim and (info.points_count or 0) == len(self._ids):
                return
            self._qdrant.delete_collection(self.COLLECTION)
        self._qdrant.create_collection(
            collection_name=self.COLLECTION,
            vectors_config=qm.VectorParams(size=self._embedder.dim, distance=qm.Distance.COSINE),
        )
        batch = 64
        for start in range(0, len(self._ids), batch):
            ids = self._ids[start : start + batch]
            vectors = self._embedder.embed([self._chunks[i].content for i in ids])
            self._qdrant.upsert(
                collection_name=self.COLLECTION,
                points=[
                    qm.PointStruct(id=self._point_id(i), vector=v, payload={"chunk_id": i})
                    for i, v in zip(ids, vectors, strict=True)
                ],
            )

    def get_chunk_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        """Return the chunk with this deterministic id, or None if it does not exist."""
        return self._chunks.get(chunk_id)

    def _dense_ranking(self, query: str, pool: int) -> list[str]:
        vector = self._embedder.embed([query])[0]
        hits = self._qdrant.query_points(
            collection_name=self.COLLECTION, query=vector, limit=pool
        ).points
        return [str((h.payload or {})["chunk_id"]) for h in hits]

    def _sparse_ranking(self, query: str, pool: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)[:pool]
        return [self._ids[i] for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        date_filter: str | None = None,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Run search and return up to ``top_k`` chunks, each with a ``score``.

        ``mode`` selects the rankers and ``rerank`` toggles the reranker (both for ablations).
        """
        if not self._ids or top_k <= 0:
            return []
        pool = min(len(self._ids), max(50, top_k * 10))
        rankings: list[list[str]] = []
        if mode in ("hybrid", "dense"):
            rankings.append(self._dense_ranking(query, pool))
        if mode in ("hybrid", "sparse"):
            rankings.append(self._sparse_ranking(query, pool))
        fused: dict[str, float] = {}
        for ranking in rankings:
            kept = [i for i in ranking if date_matches(self._chunks[i].published_date, date_filter)]
            for rank, chunk_id in enumerate(kept, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        candidates = sorted(fused, key=fused.__getitem__, reverse=True)
        if self._reranker is not None and rerank:
            candidates = candidates[: max(top_k * 4, 20)]
            cand_chunks = [self._chunks[i] for i in candidates]
            rerank_scores = self._reranker.score(query, cand_chunks)
            ranked = sorted(zip(candidates, rerank_scores, strict=True), key=lambda p: p[1], reverse=True)
        else:
            ranked = [(i, fused[i]) for i in candidates]

        return [
            self._chunks[i].model_copy(update={"score": float(score)}) for i, score in ranked[:top_k]
        ]
