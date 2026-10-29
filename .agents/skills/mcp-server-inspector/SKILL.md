---
name: mcp-server-inspector
description: >-
  Use this skill when the user wants to test, inspect, or validate the FastMCP
  servers (rag_server.py and memory_server.py) either via Python smoke tests
  or the MCP Inspector UI (@modelcontextprotocol/inspector).
---

# FastMCP Server Validation & Inspector Guide

Use this skill to verify that `src/mcp_servers/rag_server.py` and `src/mcp_servers/memory_server.py` expose valid MCP tools and return Pydantic-validated JSON payloads.

## Execution Steps

1. **Automated Smoke Test (Headless):**
   Run the MCP tool registration and invocation tests via `pytest`:
   ```powershell
   uv run pytest tests/test_phase1_memory_and_mcp.py -v
   ```
2. **Interactive MCP Inspector Web UI (Optional when requested by user):**
   Launch the MCP Inspector against either server:
   ```powershell
   npx @modelcontextprotocol/inspector uv run python -m src.mcp_servers.memory_server
   ```
   or:
   ```powershell
   npx @modelcontextprotocol/inspector uv run python -m src.mcp_servers.rag_server
   ```

