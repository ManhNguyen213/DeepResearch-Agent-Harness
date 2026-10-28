# Deep Research Agent Harness (GR2)

**Design and Implementation of an Agentic Harness with Self-Evolving Multi-Tier Cognitive Memory for Autonomous Deep Research**

---

## Architecture Overview (4-Layer Design)

1. **Layer 1 – Interactive UI (`src/ui/`):**
   - Built with **Chainlit** to stream real-time reasoning steps, sub-agent tool traces, token budget utilization, and memory state diffs.
2. **Layer 2 – Agent Harness Core Runtime (`src/harness/`):**
   - **Context & Budget Controller (`compactor.py`):** Tracks token usage, injects `Procedural Memory` rules at session start, and performs **Auto-Compaction** into a `StateSnapshot` when Working Memory reaches 70% of the token budget.
   - **Planner (`planner.py`):** Decomposes complex multi-hop research questions into structured `SubQueryTask` nodes.
   - **Isolated Sub-agent Workers (`subagent.py`):** Executes **Context Forking** in isolated message buffers to read raw retrieved chunks and returns only compact `StructuredEvidenceCard` objects (~300–500 tokens).
   - **Verification & Critic Gate (`verifier.py`):** Deterministic `source_chunk_id` check + semantic entailment verification to prevent **Memory Poisoning**.
   - **Orchestrator (`graph.py`):** Explicit cyclic control flow built on **LangGraph (`StateGraph`)**.
3. **Layer 3 – MCP Tools & 3-Tier Cognitive Memory Engine (`src/memory/` & `src/mcp_servers/`):**
   - **Tier 1 (Working Memory):** Active session state, scratchpad, and hypotheses.
   - **Tier 2 (Temporal Semantic Memory - `temporal_store.py`):** Stores verified facts with temporal validity (`valid_from`, `valid_to`, `status: ACTIVE | SUPERSEDED`) and non-destructive **Temporal Conflict Resolution**.
   - **Tier 3 (Episodic & Procedural Memory - `episodic_store.py`):** Stores past research traces and reusable query strategies.
   - **FastMCP Servers (`rag_server.py`, `memory_server.py`):** Standardized Model Context Protocol servers for Hybrid RAG (Qdrant Dense + BM25 + Reranker) and Memory Read/Write operations.
4. **Layer 4 – Observability & Evaluation (`src/observability/` & `tests/`):**
   - Telemetry and token/latency tracing powered by **Langfuse**.

---

## Quick Start

```powershell
# 1. Create virtual environment and install dependencies via uv
uv sync --all-groups

# 2. Configure environment variables
Copy-Item .env.example .env
# Edit .env with your GEMINI_API_KEY and LANGFUSE keys

# 3. Run unit tests
uv run pytest -v
```
