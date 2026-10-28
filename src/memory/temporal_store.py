"""Tier 2: Temporal Semantic Memory with Non-Destructive Conflict Resolution (SQLite + Qdrant)."""

import sqlite3
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.memory.embeddings import Embedder
from src.schemas.memory import MemoryStatus, TemporalFactRecord


class TemporalSemanticStore:
    """Manages verified temporal facts and executes non-destructive Temporal Conflict Resolution.

    Design Rationale:
        Traditional vector memories overwrite or duplicate conflicting facts across time periods.
        This store maintains structured temporal validity `(valid_from, valid_to, status)`.
        When a newer verified fact `(entity, attribute, value_new, T_new)` arrives where
        `T_new > T_old`, the older `ACTIVE` fact is never physically deleted; instead its
        `valid_to` is set to `T_new` and `status` transitions to `SUPERSEDED`.
    """

    VECTOR_COLLECTION = "semantic_memory_vectors"

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        qdrant_client: QdrantClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialise the SQLite relational side and, optionally, the Qdrant vector side.

        Data Flow:
            commit_fact -> SQLite (exact temporal filtering) + Qdrant (semantic similarity).
            Both sides are updated in the same call; Qdrant payload ``status`` mirrors SQLite.

        Args:
            db_path: SQLite file path or ``":memory:"``.
            qdrant_client: Qdrant client. If None, semantic search is disabled.
            embedder: Embedder used to vectorise facts. Required when ``qdrant_client`` is set.
        """
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        if qdrant_client is not None and embedder is None:
            raise ValueError("An embedder is required when a Qdrant client is provided.")
        self._qdrant = qdrant_client
        self._embedder = embedder
        if self._qdrant is not None and self._embedder is not None:
            self._init_vector_collection()

    def _init_vector_collection(self) -> None:
        assert self._qdrant is not None and self._embedder is not None
        if not self._qdrant.collection_exists(self.VECTOR_COLLECTION):
            self._qdrant.create_collection(
                collection_name=self.VECTOR_COLLECTION,
                vectors_config=qm.VectorParams(size=self._embedder.dim, distance=qm.Distance.COSINE),
            )

    @staticmethod
    def _point_id(memory_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))

    @staticmethod
    def _fact_text(fact: TemporalFactRecord) -> str:
        return f"{fact.entity} {fact.attribute}: {fact.value}"

    def _index_fact(self, fact: TemporalFactRecord) -> None:
        if self._qdrant is None or self._embedder is None:
            return
        vector = self._embedder.embed([self._fact_text(fact)])[0]
        self._qdrant.upsert(
            collection_name=self.VECTOR_COLLECTION,
            points=[
                qm.PointStruct(
                    id=self._point_id(fact.memory_id),
                    vector=vector,
                    payload={
                        "memory_id": fact.memory_id,
                        "entity": fact.entity,
                        "attribute": fact.attribute,
                        "status": fact.status.value,
                    },
                )
            ],
        )

    def _mark_vector_superseded(self, memory_id: str) -> None:
        if self._qdrant is None:
            return
        self._qdrant.set_payload(
            collection_name=self.VECTOR_COLLECTION,
            payload={"status": MemoryStatus.SUPERSEDED.value},
            points=[self._point_id(memory_id)],
        )

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        active_only: bool = True,
    ) -> list[tuple[TemporalFactRecord, float]]:
        """Semantic similarity search over facts; returns ``(record, cosine_score)`` pairs."""
        if self._qdrant is None or self._embedder is None:
            raise RuntimeError("Semantic search requires a Qdrant client and an embedder.")
        query_filter = None
        if active_only:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="status", match=qm.MatchValue(value=MemoryStatus.ACTIVE.value)
                    )
                ]
            )
        vector = self._embedder.embed([query])[0]
        hits = self._qdrant.query_points(
            collection_name=self.VECTOR_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
        ).points
        results: list[tuple[TemporalFactRecord, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            record = self.get_fact_by_id(str(payload.get("memory_id", "")))
            if record is not None:
                results.append((record, float(hit.score)))
        return results

    def get_fact_by_id(self, memory_id: str) -> TemporalFactRecord | None:
        """Fetch a single fact by ``memory_id`` (any status)."""
        row = self._conn.execute(
            "SELECT * FROM temporal_facts WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TemporalFactRecord:
        return TemporalFactRecord(
            memory_id=row["memory_id"],
            entity=row["entity"],
            attribute=row["attribute"],
            value=row["value"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            status=MemoryStatus(row["status"]),
            source_chunk_id=row["source_chunk_id"],
            confidence_score=float(row["confidence_score"]),
            created_at=row["created_at"],
        )

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_facts (
                    memory_id TEXT PRIMARY KEY,
                    entity TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    status TEXT NOT NULL,
                    source_chunk_id TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entity_attr_status
                ON temporal_facts (entity, attribute, status)
                """
            )

    def commit_fact(self, new_fact: TemporalFactRecord) -> TemporalFactRecord:
        """Insert a verified fact while resolving temporal conflicts against existing ACTIVE facts."""
        active_records = self.get_facts(
            entity=new_fact.entity,
            attribute=new_fact.attribute,
            active_only=True,
        )

        superseded_ids: list[str] = []
        with self._conn:
            for existing in active_records:
                if existing.value.strip() == new_fact.value.strip():
                    return existing

                if new_fact.valid_from >= existing.valid_from:
                    # Non-destructive conflict resolution: supersede the older record
                    self._conn.execute(
                        """
                        UPDATE temporal_facts
                        SET valid_to = ?, status = ?
                        WHERE memory_id = ?
                        """,
                        (new_fact.valid_from, MemoryStatus.SUPERSEDED.value, existing.memory_id),
                    )
                    superseded_ids.append(existing.memory_id)
                else:
                    # Out-of-order historical insertion: mark the newly inserted historical fact as superseded
                    new_fact = new_fact.model_copy(
                        update={
                            "valid_to": existing.valid_from,
                            "status": MemoryStatus.SUPERSEDED,
                        }
                    )

            self._conn.execute(
                """
                INSERT INTO temporal_facts (
                    memory_id, entity, attribute, value,
                    valid_from, valid_to, status,
                    source_chunk_id, confidence_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_fact.memory_id,
                    new_fact.entity,
                    new_fact.attribute,
                    new_fact.value,
                    new_fact.valid_from,
                    new_fact.valid_to,
                    new_fact.status.value,
                    new_fact.source_chunk_id,
                    new_fact.confidence_score,
                    new_fact.created_at,
                ),
            )
        for old_id in superseded_ids:
            self._mark_vector_superseded(old_id)
        self._index_fact(new_fact)
        return new_fact

    def get_facts(
        self,
        entity: str,
        attribute: str | None = None,
        active_only: bool = True,
    ) -> list[TemporalFactRecord]:
        """Query temporal facts for an entity, optionally filtering by attribute and ACTIVE status."""
        query = "SELECT * FROM temporal_facts WHERE entity = ?"
        params: list[str] = [entity]

        if attribute is not None:
            query += " AND attribute = ?"
            params.append(attribute)

        if active_only:
            query += " AND status = ?"
            params.append(MemoryStatus.ACTIVE.value)

        query += " ORDER BY valid_from DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            TemporalFactRecord(
                memory_id=row["memory_id"],
                entity=row["entity"],
                attribute=row["attribute"],
                value=row["value"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                status=MemoryStatus(row["status"]),
                source_chunk_id=row["source_chunk_id"],
                confidence_score=float(row["confidence_score"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
