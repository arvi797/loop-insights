"""SQLite-backed cache for computed insights.

Computing a report fans out into many GitHub calls (PRs + reviews per PR), so we
persist the result keyed by (repo, window) with a TTL. Repeat queries for the same
window are served from SQLite instead of re-hitting the upstream API — which is the
performance property the assignment asks us to demonstrate.

The narrative is cached alongside its report so the LLM isn't re-invoked needlessly.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.models import CollaborationHealth, Narrative

DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour


class InsightCache:
    def __init__(self, path: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._path = path
        self._ttl = ttl_seconds
        self._in_memory = path == ":memory:"
        # An in-memory DB lives only as long as its connection, so hold one open for
        # the cache's lifetime. File-backed stores open a connection per operation,
        # which keeps things simple and thread-safe across requests.
        self._shared_conn: sqlite3.Connection | None = None
        if self._in_memory:
            self._shared_conn = sqlite3.connect(self._path, check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, closing it afterwards unless it's the shared one."""
        if self._shared_conn is not None:
            yield self._shared_conn
            return
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    cache_key TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    health_json TEXT NOT NULL,
                    narrative_json TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def key(repo: str, period_start: str, period_end: str) -> str:
        return f"{repo}|{period_start}|{period_end}"

    def get_health(self, cache_key: str) -> CollaborationHealth | None:
        row = self._fresh_row(cache_key)
        if row is None:
            return None
        return CollaborationHealth.model_validate_json(row["health_json"])

    def get_narrative(self, cache_key: str) -> Narrative | None:
        row = self._fresh_row(cache_key)
        if row is None or row["narrative_json"] is None:
            return None
        return Narrative.model_validate_json(row["narrative_json"])

    def put_health(self, cache_key: str, health: CollaborationHealth) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO insights (cache_key, repo, health_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    health_json=excluded.health_json,
                    created_at=excluded.created_at,
                    narrative_json=NULL
                """,
                (cache_key, health.repo, health.model_dump_json(), time.time()),
            )
            conn.commit()

    def put_narrative(self, cache_key: str, narrative: Narrative) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE insights SET narrative_json=? WHERE cache_key=?",
                (narrative.model_dump_json(), cache_key),
            )
            conn.commit()

    def _fresh_row(self, cache_key: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM insights WHERE cache_key=?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        if time.time() - row["created_at"] > self._ttl:
            return None  # stale; caller recomputes and overwrites
        return row
