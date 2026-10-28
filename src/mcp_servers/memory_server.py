"""FastMCP Server exposing 3-Tier Cognitive Memory Read/Write tools.

Data Flow:
    MCP client -> tool -> TemporalSemanticStore (SQLite + Qdrant) / EpisodicProceduralStore (SQLite)
    -> Pydantic ``model_dump()`` -> JSON.

Design Rationale:
    Memory is exposed *only* through MCP tools so the Harness never touches storage directly and
    every read/write is a standardised, traceable tool call. Stores are lazily created singletons.
    Write access (``commit_temporal_memory``) is meant to be called only after the Verification
    Gate (Phase 3) has approved a claim; the server itself enforces schema validity.
"""

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from src.config import get_settings
from src.memory.embeddings import get_embedder
from src.memory.episodic_store import EpisodicProceduralStore
from src.memory.temporal_store import TemporalSemanticStore
from src.schemas.memory import EpisodicLogRecord, TemporalFactRecord

mcp = FastMCP("DeepResearch-CognitiveMemory-Server")
_store: TemporalSemanticStore | None = None
_episodic: EpisodicProceduralStore | None = None


def get_memory_store() -> TemporalSemanticStore:
    """Lazy-initialize the TemporalSemanticStore singleton (SQLite + Qdrant)."""
    global _store
    if _store is None:
        settings = get_settings()
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            path = settings.qdrant_local_path / "semantic_memory"
            path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(path))
        _store = TemporalSemanticStore(
            db_path=settings.sqlite_db_path,
            qdrant_client=client,
            embedder=get_embedder(settings),
        )
    return _store


def get_episodic_store() -> EpisodicProceduralStore:
    """Lazy-initialize the EpisodicProceduralStore singleton."""
    global _episodic
    if _episodic is None:
        _episodic = EpisodicProceduralStore(db_path=get_settings().sqlite_db_path)
    return _episodic


@mcp.tool()
def query_temporal_memory(
    entity: str,
    attribute: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Query Temporal Semantic Memory for verified facts about an entity.

    Args:
        entity: Canonical entity name (e.g., 'FPT_Corporation').
        attribute: Optional attribute filter (e.g., 'AI_Capex_Investment').
        active_only: If True, returns only ACTIVE records; if False, includes SUPERSEDED history.
    """
    store = get_memory_store()
    records = store.get_facts(entity=entity, attribute=attribute, active_only=active_only)
    return [r.model_dump() for r in records]


@mcp.tool()
def search_temporal_memory(query: str, top_k: int = 5, active_only: bool = True) -> list[dict]:
    """Semantic (vector) search over Temporal Semantic Memory.

    Returns facts as dictionaries with an extra ``score`` (cosine similarity) field.
    """
    hits = get_memory_store().search_similar(query=query, top_k=top_k, active_only=active_only)
    return [{**record.model_dump(), "score": score} for record, score in hits]


@mcp.tool()
def commit_temporal_memory(
    entity: str,
    attribute: str,
    value: str,
    valid_from: str,
    source_chunk_id: str,
    confidence_score: float = 0.90,
) -> dict:
    """Commit a verified fact to Temporal Semantic Memory with automatic conflict resolution."""
    store = get_memory_store()
    record = TemporalFactRecord(
        entity=entity,
        attribute=attribute,
        value=value,
        valid_from=valid_from,
        source_chunk_id=source_chunk_id,
        confidence_score=confidence_score,
    )
    committed = store.commit_fact(record)
    return committed.model_dump()


@mcp.tool()
def get_procedural_rules(query: str = "", limit: int = 5) -> list[dict]:
    """Return Procedural Memory rules whose ``domain_pattern`` matches the query.

    With an empty query, returns the globally most effective rules (used at session start).
    """
    rules = get_episodic_store().search_procedural_rules(query=query, limit=limit)
    return [r.model_dump() for r in rules]


@mcp.tool()
def record_episode(
    session_id: str,
    user_query: str,
    decomposed_sub_queries: list[str] | None = None,
    verified_chunk_ids: list[str] | None = None,
    total_tokens_used: int = 0,
    success: bool = True,
    lessons_learned: str = "",
) -> dict:
    """Persist an Episodic Memory record for a completed research session."""
    episode = EpisodicLogRecord(
        session_id=session_id,
        user_query=user_query,
        decomposed_sub_queries=decomposed_sub_queries or [],
        verified_chunk_ids=verified_chunk_ids or [],
        total_tokens_used=total_tokens_used,
        success=success,
        lessons_learned=lessons_learned,
    )
    get_episodic_store().save_episode(episode)
    return episode.model_dump()


if __name__ == "__main__":
    mcp.run()
