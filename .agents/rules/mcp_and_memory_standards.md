# MCP Tooling & 3-Tier Cognitive Memory Standards

## 1. FastMCP Tool Design
* All external data retrieval and memory operations (`hybrid_search`, `web_search`, `query_temporal_memory`, `commit_temporal_memory`, `retrieve_procedural_rules`) must be exposed through **FastMCP** (`mcp.server.fastmcp.FastMCP`).
* Every `@mcp.tool()` function must:
  * Include explicit Python 3.11+ type annotations and a clear docstring describing parameters and return structure.
  * Return deterministic JSON-serializable structures (validated via `Pydantic v2` models) containing `source_chunk_id` and temporal metadata (`valid_from`, `valid_to`).

## 2. Anti-Poisoning Verification Gate
* No fact may be committed to `TemporalSemanticMemory` unless it passes both checks in `VerificationGate`:
  1. **Deterministic Citation Check:** `source_chunk_id` must exist in the set of chunk IDs actually returned by MCP tools during the current session.
  2. **Semantic Entailment / Groundedness Check:** The extracted claim must be entailed by the source chunk text with `confidence_score >= grounding_threshold` (default `0.80`).

## 3. Non-Destructive Temporal Conflict Resolution
* When inserting a new verified fact `(entity, attribute, value_new, valid_from=T_new)`:
  * Query existing `ACTIVE` records matching `(entity, attribute)`.
  * If a conflicting or updated record exists with `T_new > T_old`, **never delete** the old record. Update the old record with `valid_to = T_new` and `status = MemoryStatus.SUPERSEDED`, then insert the new record with `status = MemoryStatus.ACTIVE` and `valid_to = None`.
