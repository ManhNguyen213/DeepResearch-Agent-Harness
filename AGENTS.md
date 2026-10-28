# AGENTS.md — Architectural Rules & Project Context (GR2)

## 1. Project Context
* **Project Title:** Design and Implementation of an Agentic Harness with Self-Evolving Multi-Tier Cognitive Memory for Autonomous Deep Research.
* **Reference Documents in Workspace (`Plan/` directory):**
  * Project Proposal (v2.0): `Plan/01_Project_Proposal_VI.md`
  * Master Roadmap & Actionable Checklist (Phase 0 – Phase 5): `Plan/02_Master_Roadmap_And_Checklist_VI.md`
  * Architecture Study Guide & Phase-by-Phase Learning Path: `Plan/03_Study_Guide_And_Learning_Path_VI.md`
  * GR2 Report Document: `Plan/Bao_Cao_Ke_Hoach_GR2.doc` (and generator `Plan/generate_docx.py`)
  * Initial Design Brain/Transcript ID: `28aa03f2-659e-4172-b0ba-ec04f697b6a2`

---

## 2. Mandatory Architectural Principles

When writing code, designing modules, or debugging in this repository, the Agent **MUST** strictly adhere to the following principles:

1. **No Black-Box Agent or Memory Wrappers:**
   * **DO NOT** use legacy pre-packaged agent wrappers (e.g., classic LangChain `initialize_agent`, `AgentExecutor`) or black-box third-party memory libraries (e.g., `Mem0`) for the core research system.
   * Use **LangGraph (`StateGraph`)** with explicit, transparent control flow to implement the **Agent Harness** and the **Custom 3-Tier Cognitive Memory Engine** from scratch.

2. **4-Layer System Architecture:**
   * **Layer 1 – UI Layer (`src/ui/`):**
     * `Chainlit` web interface displaying real-time **Streaming Thought Process** and a **Memory & Context Inspector** (token budget status and memory diffs).
   * **Layer 2 – Agent Harness / Core Runtime (`src/harness/`):**
     * `Context & Budget Controller`: Monitors token consumption, triggers **Auto-Compaction** when working context exceeds the configured budget threshold (e.g., 70% token budget), and injects `Procedural Memory` rules at session initialization.
     * `Planner`: Decomposes complex multi-hop research queries into structured sub-tasks (`SubQueryTask`).
     * `Isolated Sub-agent Workers`: Executes **Context Forking (Sub-agent Context Isolation)** — each Sub-agent runs in an isolated `messages` state to read and filter raw retrieved chunks, discarding raw text after extraction and returning only a compact `StructuredEvidenceCard` (~300–500 tokens, defined via `Pydantic v2`) to the Main Agent.
     * `Verification & Critic Gate`: Performs deterministic citation validation (`source_chunk_id` existence check) and semantic groundedness/entailment verification before report synthesis or long-term memory commits (preventing **Memory Poisoning**).
   * **Layer 3 – MCP Tools & 3-Tier Memory Engine (`src/mcp_servers/` & `src/memory/`):**
     * All data retrieval and memory tools (Hybrid RAG inherited from GR1 using Qdrant Dense + BM25 + Cross-Encoder Reranker, Web Search, and Memory Read/Write) must be standardized via **MCP (Model Context Protocol - `FastMCP`)**.
     * **3-Tier Cognitive Memory:**
       1. *Working Memory:* Current session state, active hypotheses, and scratchpad within LangGraph state.
       2. *Temporal Semantic Memory (SQLite/PostgreSQL + Qdrant):* Stores verified facts using the schema `(memory_id, entity, attribute, value, valid_from, valid_to, status [ACTIVE | SUPERSEDED], source_chunk_id, confidence_score)`. Implements **Temporal Conflict Resolution**: never physically delete outdated facts; instead update `valid_to = T_new`, set `status = SUPERSEDED`, and insert the new fact as `ACTIVE`.
       3. *Episodic & Procedural Memory:* Logs past execution traces and distills successful search/reasoning trajectories into reusable query strategies (`ProceduralRule`).
   * **Layer 4 – Observability & Evaluation (`src/observability/` & `tests/`):**
     * Tracks token usage, node latency, and execution traces via **Langfuse**.
     * Evaluates across 3 dimensions: *Multi-hop QA Quality*, *Multi-session Memory Efficacy*, and *Harness Engineering Metrics*.

---

## 3. Coding Standards & Conventions
* **Primary Programming Language:** **English** must be used for all source code, module/variable/function/class names, type annotations, docstrings, code comments, LLM prompts, log messages, and Git commits. (Vietnamese may be used when conversing with the user or writing GR2 academic report explanations).
* **Python Version & Typing:** Python 3.11+, using comprehensive Type Hints (`typing`) and clear Google-style docstrings.
* **Structured Outputs & Validation:** Always use `Pydantic v2` (`BaseModel`, `Field`, `ConfigDict`) for all LangGraph States, Evidence Cards, Memory Records, and structured LLM outputs.
* **Configuration & Security:** Never hardcode API keys or secrets; always load them from environment variables via `pydantic-settings` / `python-dotenv` (`.env`).
* **Architectural Clarity:** When implementing a new module, include a concise explanation of its **Data Flow** and **Design Rationale** to support the user's GR2 thesis documentation and defense.
