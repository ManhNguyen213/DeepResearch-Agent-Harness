"""Pytest configuration: keep the whole suite hermetic.

* Langfuse OpenTelemetry export is disabled so tests never touch the network.
* ``LLM_ENABLED=false`` guarantees no test can reach the Gemini API, even when a real key is
  present in ``.env``. Tests that exercise LLM code paths pass ``use_llm=True`` explicitly and
  monkeypatch the gateway with a fake (see ``tests/test_phase2_llm_paths.py``).
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
os.environ["LLM_ENABLED"] = "false"

CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "corpus"


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Offline environment: HashingEmbedder, temp SQLite + Qdrant, fixture corpus, no LLM."""
    from src.config import get_settings
    from src.mcp_servers import memory_server, rag_server

    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("EMBEDDING_DIM", "128")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("RAG_CORPUS_PATH", str(CORPUS_DIR))
    monkeypatch.setenv("RERANKER_MODEL", "")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(memory_server, "_store", None)
    monkeypatch.setattr(memory_server, "_episodic", None)
    monkeypatch.setattr(rag_server, "_retriever", None)
    yield tmp_path
    if memory_server._store is not None and memory_server._store._qdrant is not None:
        memory_server._store._qdrant.close()
    if rag_server._retriever is not None:
        rag_server._retriever._qdrant.close()
    get_settings.cache_clear()

