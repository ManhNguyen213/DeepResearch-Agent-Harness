"""Phase 1 tests: Temporal Semantic Store (SQLite + Qdrant), Episodic/Procedural store,
Hybrid retriever, and the FastMCP servers. All tests run offline with the hashing embedder."""

import json
import time
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.memory.embeddings import HashingEmbedder
from src.memory.episodic_store import EpisodicProceduralStore
from src.memory.temporal_store import TemporalSemanticStore
from src.retrieval.hybrid_retriever import HybridRetriever, date_matches, load_chunks_from_dir
from src.schemas.memory import EpisodicLogRecord, MemoryStatus, ProceduralRule, TemporalFactRecord

CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "corpus"


def _fact(mid: str, value: str, valid_from: str) -> TemporalFactRecord:
    return TemporalFactRecord(
        memory_id=mid,
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        value=value,
        valid_from=valid_from,
        source_chunk_id=f"doc_{mid}",
        confidence_score=0.95,
    )


@pytest.fixture
def store() -> TemporalSemanticStore:
    return TemporalSemanticStore(
        db_path=":memory:",
        qdrant_client=QdrantClient(":memory:"),
        embedder=HashingEmbedder(dim=128),
    )


def test_semantic_index_mirrors_supersede(store: TemporalSemanticStore) -> None:
    store.commit_fact(_fact("m24", "Invested 100M USD in AI infrastructure", "2024-03-01"))
    store.commit_fact(_fact("m25", "Expanded AI investment to 200M USD", "2025-06-01"))

    active = store.search_similar("AI investment", top_k=5, active_only=True)
    assert [r.memory_id for r, _ in active] == ["m25"]

    everything = store.search_similar("AI investment", top_k=5, active_only=False)
    by_id = {r.memory_id: r for r, _ in everything}
    assert set(by_id) == {"m24", "m25"}
    assert by_id["m24"].status == MemoryStatus.SUPERSEDED
    assert by_id["m24"].valid_to == "2025-06-01"


def test_out_of_order_insertion_keeps_newest_active(store: TemporalSemanticStore) -> None:
    store.commit_fact(_fact("m25", "Expanded AI investment to 200M USD", "2025-06-01"))
    old = store.commit_fact(_fact("m24", "Invested 100M USD in AI infrastructure", "2024-03-01"))
    assert old.status == MemoryStatus.SUPERSEDED
    assert old.valid_to == "2025-06-01"
    active = store.get_facts("FPT_Corporation", "AI_Capex_Investment")
    assert [f.memory_id for f in active] == ["m25"]


def test_temporal_query_latency_under_50ms(store: TemporalSemanticStore) -> None:
    for i in range(200):
        store.commit_fact(
            TemporalFactRecord(
                entity=f"E{i}",
                attribute="A",
                value=f"value {i}",
                valid_from="2025-01-01",
                source_chunk_id=f"c{i}",
                confidence_score=0.9,
            )
        )
    start = time.perf_counter()
    store.get_facts("E100", "A")
    assert (time.perf_counter() - start) * 1000 < 50
    start = time.perf_counter()
    store.search_similar("value 100", top_k=5)
    assert (time.perf_counter() - start) * 1000 < 50


def test_episodic_and_procedural_store() -> None:
    ep = EpisodicProceduralStore()
    ep.save_episode(EpisodicLogRecord(episode_id="e1", session_id="s1", user_query="q1", success=True))
    ep.save_episode(EpisodicLogRecord(episode_id="e2", session_id="s2", user_query="q2", success=False))
    assert [e.episode_id for e in ep.list_episodes(session_id="s1")] == ["e1"]
    assert len(ep.list_episodes()) == 2

    ep.save_procedural_rule(
        ProceduralRule(
            rule_id="r1",
            domain_pattern="financial_capex_comparison",
            strategy_instruction="Search each year's report separately with date_filter.",
            effectiveness_score=0.9,
        )
    )
    ep.save_procedural_rule(
        ProceduralRule(rule_id="r2", domain_pattern="legal_contract_review", strategy_instruction="x")
    )
    hits = ep.search_procedural_rules("Compare capex of FPT and VNG")
    assert [r.rule_id for r in hits] == ["r1"]
    ep.increment_rule_usage("r1")
    assert ep.list_procedural_rules()[0].usage_count == 1 or ep.search_procedural_rules("capex")[0].usage_count == 1


def test_date_matches() -> None:
    assert date_matches("2025-06-01", "2025")
    assert date_matches("2025-06-01", "2025-06")
    assert not date_matches("2024-03-01", "2025")
    assert date_matches("2025-06-01", "2025-01-01..2025-12-31")
    assert date_matches("2025-06-01", "..2025-06-01")
    assert not date_matches(None, "2025")
    assert date_matches(None, None)


@pytest.fixture
def retriever() -> HybridRetriever:
    return HybridRetriever(
        chunks=load_chunks_from_dir(CORPUS_DIR),
        qdrant_client=QdrantClient(":memory:"),
        embedder=HashingEmbedder(dim=256),
    )


def test_hybrid_search_returns_chunk_ids(retriever: HybridRetriever) -> None:
    results = retriever.search("FPT AI capex investment", top_k=3)
    assert results and results[0].chunk_id.startswith("doc_fpt")
    assert all(r.chunk_id and r.content and r.score > 0 for r in results)
    assert len(results) <= 3


def test_hybrid_search_date_filter_and_get_by_id(retriever: HybridRetriever) -> None:
    results = retriever.search("FPT AI investment", top_k=5, date_filter="2024")
    assert [r.chunk_id for r in results] == ["doc_fpt_2024_q1_p05"]
    assert retriever.get_chunk_by_id("doc_fpt_2025_q2_p14") is not None
    assert retriever.get_chunk_by_id("doc_nonexistent") is None


def test_mcp_servers_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.config import get_settings
    from src.mcp_servers import memory_server, rag_server

    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("EMBEDDING_DIM", "128")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("RAG_CORPUS_PATH", str(CORPUS_DIR))
    monkeypatch.setenv("RERANKER_MODEL", "")
    get_settings.cache_clear()
    monkeypatch.setattr(memory_server, "_store", None)
    monkeypatch.setattr(memory_server, "_episodic", None)
    monkeypatch.setattr(rag_server, "_retriever", None)
    try:
        hits = rag_server.hybrid_search("FPT AI capex", top_k=2)
        assert {"chunk_id", "content", "published_date", "score"} <= set(hits[0])
        assert rag_server.get_chunk_by_id(hits[0]["chunk_id"])["chunk_id"] == hits[0]["chunk_id"]
        assert rag_server.get_chunk_by_id("missing") is None

        for value, date in (("100M USD", "2024-03-01"), ("200M USD", "2025-06-01")):
            memory_server.commit_temporal_memory("FPT", "Capex", value, date, "doc_x")
        history = memory_server.query_temporal_memory("FPT", "Capex", active_only=False)
        old = next(r for r in history if r["value"] == "100M USD")
        assert old["status"] == "SUPERSEDED" and old["valid_to"] == "2025-06-01"
        assert len(memory_server.query_temporal_memory("FPT", "Capex")) == 1
        assert memory_server.search_temporal_memory("Capex 200M", top_k=1)[0]["value"] == "200M USD"

        memory_server.record_episode("s1", "q", lessons_learned="use date_filter")
        assert memory_server.get_procedural_rules("anything") == []
        json.dumps(history)  # JSON-serialisable
    finally:
        memory_server.get_memory_store()._qdrant.close()  # release local Qdrant lock
        get_settings.cache_clear()


def test_ingest_chunking_deterministic_ids(tmp_path: Path) -> None:
    from src.retrieval.ingest import chunk_text, ingest_directory

    raw = tmp_path / "raw"
    raw.mkdir()
    paragraphs = [f"Paragraph {i}. " + "word " * 60 for i in range(20)]
    (raw / "My_Doc.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")

    chunks = chunk_text("\n\n".join(paragraphs), max_chars=800, overlap_chars=100)
    assert len(chunks) > 1 and all(len(c) <= 900 for c in chunks)

    out = tmp_path / "out.jsonl"
    n1 = ingest_directory(raw, out, max_chars=800, overlap_chars=100)
    ids1 = [json.loads(line)["chunk_id"] for line in out.read_text("utf-8").splitlines()]
    n2 = ingest_directory(raw, out, max_chars=800, overlap_chars=100)
    ids2 = [json.loads(line)["chunk_id"] for line in out.read_text("utf-8").splitlines()]
    assert n1 == n2 == len(ids1) and ids1 == ids2
    assert ids1[0] == "doc_my_doc_c0000"


def test_dense_index_persisted_and_reused(tmp_path: Path) -> None:
    chunks = load_chunks_from_dir(CORPUS_DIR)
    embedder = HashingEmbedder(dim=64)
    client = QdrantClient(path=str(tmp_path / "q"))
    HybridRetriever(chunks, client, embedder)

    class Boom(HashingEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            if len(texts) > 1:
                raise AssertionError("corpus was re-embedded")
            return super().embed(texts)

    reused = HybridRetriever(chunks, client, Boom(dim=64))
    assert reused.search("FPT", top_k=1)
    client.close()
