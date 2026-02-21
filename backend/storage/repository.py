"""SQLite-backed persistence for market snapshots and agent traces."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict


class SQLiteRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_sentiment_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        captured_at TEXT NOT NULL,
                        news_date TEXT,
                        total_articles INTEGER NOT NULL,
                        sentiment_avg REAL NOT NULL,
                        positive_count INTEGER NOT NULL,
                        negative_count INTEGER NOT NULL,
                        neutral_count INTEGER NOT NULL,
                        source TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        query_model TEXT NOT NULL,
                        resolved_model TEXT NOT NULL,
                        data_date TEXT NOT NULL,
                        selected_action TEXT NOT NULL,
                        raw_action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        entropy REAL NOT NULL,
                        value REAL NOT NULL,
                        safe_mode INTEGER NOT NULL,
                        safe_reason TEXT,
                        action_probs_json TEXT NOT NULL,
                        expected_rewards_json TEXT NOT NULL,
                        news_context_json TEXT,
                        decision_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at ON agent_decisions(created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_decisions_model ON agent_decisions(resolved_model)"
                )
                conn.commit()
            self._initialized = True

    def _to_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def save_market_sentiment(self, snapshot: Dict[str, Any]) -> None:
        self.initialize()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO market_sentiment_snapshots (
                    captured_at, news_date, total_articles, sentiment_avg,
                    positive_count, negative_count, neutral_count, source, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    snapshot.get("date"),
                    int(snapshot.get("total", 0)),
                    float(snapshot.get("sentiment_avg", 0.0)),
                    int(snapshot.get("positive_count", 0)),
                    int(snapshot.get("negative_count", 0)),
                    int(snapshot.get("neutral_count", 0)),
                    str(snapshot.get("source", "unknown")),
                    self._to_json(snapshot),
                ),
            )
            conn.commit()

    def save_agent_decision(
        self,
        *,
        query_model: str,
        resolved_model: str,
        data_date: str,
        decision: Any,
        news_context: Dict[str, Any] | None = None,
    ) -> None:
        self.initialize()
        now = datetime.now(timezone.utc).isoformat()
        if is_dataclass(decision):
            decision_payload = asdict(decision)
        else:
            decision_payload = dict(decision)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_decisions (
                    created_at, query_model, resolved_model, data_date,
                    selected_action, raw_action, confidence, entropy, value,
                    safe_mode, safe_reason, action_probs_json, expected_rewards_json,
                    news_context_json, decision_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    query_model,
                    resolved_model,
                    data_date,
                    str(decision_payload.get("action", "")),
                    str(decision_payload.get("raw_action", "")),
                    float(decision_payload.get("confidence", 0.0)),
                    float(decision_payload.get("entropy", 0.0)),
                    float(decision_payload.get("value", 0.0)),
                    1 if bool(decision_payload.get("safe_mode", False)) else 0,
                    decision_payload.get("safe_reason"),
                    self._to_json(decision_payload.get("action_probs", {})),
                    self._to_json(decision_payload.get("expected_rewards", {})),
                    self._to_json(news_context or {}),
                    self._to_json(decision_payload),
                ),
            )
            conn.commit()

    def health(self) -> Dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            decision_count = conn.execute(
                "SELECT COUNT(*) AS c FROM agent_decisions"
            ).fetchone()["c"]
            sentiment_count = conn.execute(
                "SELECT COUNT(*) AS c FROM market_sentiment_snapshots"
            ).fetchone()["c"]
        return {
            "backend": "sqlite",
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "agent_decision_rows": int(decision_count),
            "sentiment_rows": int(sentiment_count),
        }
