"""Retrieval ablation on the GR1 benchmark (60 questions): dense vs. sparse vs. hybrid vs. hybrid+rerank.

Metric (inherited from GR1 ``compute_retrieval_metrics`` fallback): *entity coverage* = fraction of
the question's gold entities that appear in the concatenated top-k retrieved text. GR1's dataset has
no gold source-document labels, so this is a proxy until Phase 4 adds labelled chunk ids.

Usage:
    uv run --no-sync python scripts/eval_retrieval.py [--top-k 5] [--no-rerank]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import get_settings
from src.memory.embeddings import get_embedder
from src.retrieval.hybrid_retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    load_chunks_from_dir,
)

DATASET = Path("data/benchmark/gr1_dataset.json")


def entity_coverage(texts: list[str], entities: list[str]) -> float:
    """Fraction of gold entities found (case-insensitive substring) in the retrieved texts."""
    if not entities:
        return 1.0
    blob = " ".join(texts).lower()
    return sum(e.lower().strip() in blob for e in entities) / len(entities)


def main() -> None:
    """Run the ablation and print a table plus per-type breakdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Include Cross-Encoder rerank row")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks_from_dir(settings.rag_corpus_path)
    qdrant_path = settings.qdrant_local_path / "documents"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    reranker = None
    rerank_model = settings.reranker_model or (
        "cross-encoder/ms-marco-MiniLM-L-6-v2" if args.rerank else None
    )
    if rerank_model and not args.no_rerank:
        reranker = CrossEncoderReranker(rerank_model)
    retriever = HybridRetriever(
        chunks, QdrantClient(path=str(qdrant_path)), get_embedder(settings), reranker=reranker
    )

    questions = json.loads(DATASET.read_text("utf-8"))
    systems = {
        "sparse (BM25)": lambda q: retriever.search(q, args.top_k, mode="sparse", rerank=False),
        "dense": lambda q: retriever.search(q, args.top_k, mode="dense", rerank=False),
        "hybrid (RRF)": lambda q: retriever.search(q, args.top_k, mode="hybrid", rerank=False),
    }
    if reranker is not None:
        systems["hybrid + rerank"] = lambda q: retriever.search(q, args.top_k, mode="hybrid")

    print(f"{len(chunks)} chunks, {len(questions)} questions, top_k={args.top_k}")
    print(f"{'system':<18}{'coverage':>10}{'multi-hop':>11}{'single':>8}{'cross':>8}{'ms/q':>8}")
    for name, fn in systems.items():
        by_type: dict[str, list[float]] = {}
        start = time.perf_counter()
        for item in questions:
            hits = fn(item["question"])
            cov = entity_coverage([h.content for h in hits], item["entities"])
            by_type.setdefault(item["type"], []).append(cov)
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        allv = [v for vs in by_type.values() for v in vs]
        row = [statistics.mean(by_type.get(t, [0.0])) for t in ("multi-hop", "single-hop", "cross-doc")]
        print(f"{name:<18}{statistics.mean(allv):>10.3f}{row[0]:>11.3f}{row[1]:>8.3f}{row[2]:>8.3f}{ms:>8.0f}")


if __name__ == "__main__":
    main()
