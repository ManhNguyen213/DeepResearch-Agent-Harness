"""FastMCP Server exposing Hybrid RAG (Qdrant Dense + BM25 + optional Cross-Encoder Reranker) tools.

Data Flow:
    MCP client -> ``hybrid_search`` / ``get_chunk_by_id`` -> ``HybridRetriever`` over the JSONL
    corpus in ``settings.rag_corpus_path`` -> ``RetrievedChunk.model_dump()`` -> JSON.

Design Rationale:
    The server is a thin, stateless-looking facade; retrieval logic lives in
    ``src.retrieval.hybrid_retriever``. The retriever is built lazily on first call so that
    importing the module (and listing tools in the MCP Inspector) is cheap.
"""

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from src.config import get_settings
from src.memory.embeddings import get_embedder
from src.observability.tracing import observe, update_span_io
from src.retrieval.hybrid_retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    load_chunks_from_dir,
)

mcp = FastMCP("DeepResearch-HybridRAG-Server")
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Lazy-initialize the HybridRetriever singleton from the configured corpus directory."""
    global _retriever
    if _retriever is None:
        settings = get_settings()
        chunks = load_chunks_from_dir(settings.rag_corpus_path)
        qdrant_path = settings.qdrant_local_path / "documents"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
        )
        _retriever = HybridRetriever(
            chunks=chunks,
            qdrant_client=QdrantClient(path=str(qdrant_path)),
            embedder=get_embedder(settings),
            reranker=reranker,
        )
    return _retriever


@mcp.tool()
@observe(
    name="hybrid-rag-retriever",
    as_type="retriever",
    capture_input=False,
    capture_output=False,
)
def hybrid_search(query: str, top_k: int = 5, date_filter: str | None = None) -> list[dict]:
    """Execute Hybrid Search (Qdrant Dense + BM25 + Reranker) and return chunks with chunk_id.

    Args:
        query: Search query string formulated by an Isolated Sub-agent Worker.
        top_k: Maximum number of reranked document chunks to return.
        date_filter: Optional published_date filter: a prefix such as '2025' or '2025-06', or an
            inclusive range 'YYYY-MM-DD..YYYY-MM-DD' (either bound may be empty).

    Returns:
        List of serialized RetrievedChunk dictionaries containing deterministic chunk_ids.
    """
    chunks = get_retriever().search(query=query, top_k=top_k, date_filter=date_filter)
    results = [c.model_dump() for c in chunks]
    update_span_io(
        input_data={"query": query, "top_k": top_k, "date_filter": date_filter},
        output_data={
            "chunks_returned": len(results),
            "chunk_ids": [r["chunk_id"] for r in results],
            "chunks": results,
        },
        metadata={"retriever_engine": "qdrant_dense_bm25_rrf"},
    )
    return results


@mcp.tool()
def get_chunk_by_id(chunk_id: str) -> dict | None:
    """Fetch a chunk by its deterministic chunk_id (used for citation verification).

    Returns:
        Serialized RetrievedChunk, or None if the chunk_id does not exist.
    """
    chunk = get_retriever().get_chunk_by_id(chunk_id)
    return chunk.model_dump() if chunk is not None else None


if __name__ == "__main__":
    mcp.run()
