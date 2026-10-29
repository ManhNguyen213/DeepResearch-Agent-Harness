---
name: verify-memory-conflict
description: >-
  Use this skill when the user wants to test, verify, or debug the Temporal
  Semantic Memory Conflict Resolution algorithm or the Anti-Poisoning
  Verification Gate. Runs automated tests to confirm that newer facts mark
  older facts as SUPERSEDED with valid_to updated, and ungrounded facts are rejected.
---

# Verify Temporal Memory Conflict Resolution & Anti-Poisoning Gate

This skill validates two core research contributions of the GR2 project:
1. **Non-Destructive Temporal Conflict Resolution:** When a newer fact (`valid_from = T_new > T_old`) for the same `(entity, attribute)` is committed, the previous `ACTIVE` record transitions to `SUPERSEDED` with `valid_to = T_new`, and the new record is inserted as `ACTIVE`.
2. **Anti-Poisoning Verification Gate:** Any candidate fact whose `source_chunk_id` is missing from the session's retrieved chunks or whose confidence/entailment score falls below threshold is blocked from entering long-term memory.

## Execution Steps

1. Run the memory and verification test suite using `uv`:
   ```powershell
   uv run pytest tests/test_schemas_and_memory.py tests/test_phase1_memory_and_mcp.py -v
   ```
2. Verify in the test output that:
   - Historical records remain queryable with `status == "SUPERSEDED"` and `valid_to == T_new`.
   - Active queries only return `status == "ACTIVE"` records by default.
   - Hallucinated `source_chunk_id` entries raise/return verification failure and are not persisted.

