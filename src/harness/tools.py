"""Tool gateway: the single entry point the Harness uses for retrieval and memory tools.

Data Flow:
    Harness node (planner / sub-agent / compactor) -> ``tools.<function>`` -> the FastMCP tool
    function registered in ``src.mcp_servers`` -> store / retriever -> JSON-serialisable dicts.

Design Rationale:
    The retrieval and memory tools are *defined* as FastMCP tools (``@mcp.tool()``), and the
    same functions are served over the MCP protocol by ``rag_server`` / ``memory_server``.
    The Harness calls them **in-process** through this module instead of spawning stdio MCP
    subprocesses: this avoids one process (and one embedding-model load) per server and keeps
    the graph deterministic and testable.

    Known limitation (to be stated in the GR2 report): in-process calls skip the MCP wire
    protocol. Keeping every call behind this gateway means switching to a real MCP client later
    only requires changing this one module, not the Harness nodes.
"""

from typing import Any


def hybrid_search(
    query: str, top_k: int = 5, date_filter: str | None = None
) -> list[dict[str, Any]]:
    """Hybrid RAG search (MCP tool ``hybrid_search``)."""
    from src.mcp_servers import rag_server

    return rag_server.hybrid_search(query=query, top_k=top_k, date_filter=date_filter)


def query_temporal_memory(
    entity: str, attribute: str | None = None, active_only: bool = True
) -> list[dict[str, Any]]:
    """Exact temporal-fact lookup (MCP tool ``query_temporal_memory``)."""
    from src.mcp_servers import memory_server

    return memory_server.query_temporal_memory(
        entity=entity, attribute=attribute, active_only=active_only
    )


def search_temporal_memory(
    query: str, top_k: int = 5, active_only: bool = True
) -> list[dict[str, Any]]:
    """Semantic temporal-fact search (MCP tool ``search_temporal_memory``)."""
    from src.mcp_servers import memory_server

    return memory_server.search_temporal_memory(query=query, top_k=top_k, active_only=active_only)


def get_procedural_rules(query: str = "", limit: int = 5) -> list[dict[str, Any]]:
    """Procedural-memory rule lookup (MCP tool ``get_procedural_rules``)."""
    from src.mcp_servers import memory_server

    return memory_server.get_procedural_rules(query=query, limit=limit)
