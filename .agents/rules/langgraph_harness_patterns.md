# LangGraph Agent Harness & Context Management Rules

## 1. Sub-agent Context Isolation (Context Forking)
* Main Agent nodes (`planner`, `verifier`, `synthesizer`, `compactor`) must **never** ingest raw retrieved document chunks directly into the primary conversation history.
* Each `SubQueryTask` spawned by the `Planner` must execute in an isolated Sub-agent worker context (via LangGraph `Send` API or an isolated subgraph/worker invocation with its own private message list).
* Once the Sub-agent finishes reading raw chunks from MCP tools, it must discard the raw tool outputs and return **only** a validated `StructuredEvidenceCard` (~300–500 tokens) to the Main Agent state.

## 2. Token Budgeting & Auto-Compaction
* Every transition into the `ContextBudgetController` node must estimate the current Working Memory token count.
* When `current_tokens >= token_budget_limit * compaction_threshold` (default `0.70`), the `Auto-Compactor` must compress historical reasoning steps and intermediate tool traces into a concise `StateSnapshot` while strictly preserving all verified `source_chunk_id` citations and active hypotheses.

## 3. State Schema Discipline
* Define LangGraph state using `Pydantic v2` (`BaseModel`) or `TypedDict` with explicit reducer annotations (`Annotated[list[...], operator.add]`) so parallel sub-agent outputs merge deterministically without race conditions.
