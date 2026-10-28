"""Tier 3: Episodic & Procedural Memory Store for execution traces and reusable strategies."""

import json
import sqlite3
from pathlib import Path

from src.schemas.memory import EpisodicLogRecord, ProceduralRule


class EpisodicProceduralStore:
    """Stores past research session episodes and reusable procedural strategies."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_logs (
                    episode_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    decomposed_sub_queries TEXT NOT NULL,
                    verified_chunk_ids TEXT NOT NULL,
                    total_tokens_used INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    lessons_learned TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS procedural_rules (
                    rule_id TEXT PRIMARY KEY,
                    domain_pattern TEXT NOT NULL,
                    strategy_instruction TEXT NOT NULL,
                    usage_count INTEGER NOT NULL,
                    effectiveness_score REAL NOT NULL
                )
                """
            )

    def save_episode(self, episode: EpisodicLogRecord) -> None:
        """Persist an episodic execution record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO episodic_logs (
                    episode_id, session_id, user_query,
                    decomposed_sub_queries, verified_chunk_ids,
                    total_tokens_used, success, lessons_learned, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.session_id,
                    episode.user_query,
                    json.dumps(episode.decomposed_sub_queries),
                    json.dumps(episode.verified_chunk_ids),
                    episode.total_tokens_used,
                    int(episode.success),
                    episode.lessons_learned,
                    episode.timestamp,
                ),
            )

    def save_procedural_rule(self, rule: ProceduralRule) -> None:
        """Persist or update a distilled procedural search strategy."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO procedural_rules (
                    rule_id, domain_pattern, strategy_instruction,
                    usage_count, effectiveness_score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rule.rule_id,
                    rule.domain_pattern,
                    rule.strategy_instruction,
                    rule.usage_count,
                    rule.effectiveness_score,
                ),
            )

    def list_procedural_rules(self, limit: int = 10) -> list[ProceduralRule]:
        """Return top procedural rules ordered by effectiveness score."""
        rows = self._conn.execute(
            """
            SELECT * FROM procedural_rules
            ORDER BY effectiveness_score DESC, usage_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            ProceduralRule(
                rule_id=row["rule_id"],
                domain_pattern=row["domain_pattern"],
                strategy_instruction=row["strategy_instruction"],
                usage_count=int(row["usage_count"]),
                effectiveness_score=float(row["effectiveness_score"]),
            )
            for row in rows
        ]

    def search_procedural_rules(self, query: str, limit: int = 5) -> list[ProceduralRule]:
        """Return rules whose ``domain_pattern`` matches ``query`` (case-insensitive).

        A rule matches when its pattern is a substring of the query, or the query is a
        substring of the pattern, or they share at least one ``_``/space-delimited token.
        Results are ordered by effectiveness score, then usage count.
        """
        needle = query.strip().lower()
        if not needle:
            return self.list_procedural_rules(limit=limit)
        query_tokens = {t for t in needle.replace("_", " ").split() if len(t) > 2}
        matched: list[ProceduralRule] = []
        for rule in self.list_procedural_rules(limit=10_000):
            pattern = rule.domain_pattern.lower()
            pattern_tokens = {t for t in pattern.replace("_", " ").split() if len(t) > 2}
            if pattern in needle or needle in pattern or (query_tokens & pattern_tokens):
                matched.append(rule)
        return matched[:limit]

    def increment_rule_usage(self, rule_id: str) -> None:
        """Increase the usage counter of a procedural rule by one."""
        with self._conn:
            self._conn.execute(
                "UPDATE procedural_rules SET usage_count = usage_count + 1 WHERE rule_id = ?",
                (rule_id,),
            )

    def list_episodes(self, session_id: str | None = None, limit: int = 20) -> list[EpisodicLogRecord]:
        """Return recent episodic logs, newest first, optionally filtered by session."""
        query = "SELECT * FROM episodic_logs"
        params: list[str | int] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [
            EpisodicLogRecord(
                episode_id=row["episode_id"],
                session_id=row["session_id"],
                user_query=row["user_query"],
                decomposed_sub_queries=json.loads(row["decomposed_sub_queries"]),
                verified_chunk_ids=json.loads(row["verified_chunk_ids"]),
                total_tokens_used=int(row["total_tokens_used"]),
                success=bool(row["success"]),
                lessons_learned=row["lessons_learned"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
