---
name: harness-token-profiler
description: >-
  Use this skill when the user asks to profile token consumption, verify
  Sub-agent Context Isolation savings, or test the Auto-Compaction threshold
  (ContextBudgetController) in the LangGraph Agent Harness.
---

# Harness Token Budget & Auto-Compaction Profiler

This skill checks the **Context & Budget Controller** (`src/harness/compactor.py`) to verify that:
1. Raw tool chunks processed inside `IsolatedSubAgentWorker` are stripped before returning `StructuredEvidenceCard` to `HarnessState`.
2. When Working Memory token usage exceeds `token_budget_limit * compaction_threshold` (default 70%), `ContextBudgetController` triggers Auto-Compaction and produces a `StateSnapshot` while preserving all verified `source_chunk_id` citations.

## Execution Steps

1. Run the token budget and compaction unit tests:
   ```powershell
   uv run pytest tests/test_schemas_and_memory.py -k "compaction or budget" -v
   ```
2. Inspect the pre-compaction vs. post-compaction token counts and ensure all `verified_citations` remain intact after compaction.
