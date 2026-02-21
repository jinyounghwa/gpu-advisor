"""PostgreSQL-backed persistence for production deployments."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict


class PostgresRepository:
    def __init__(self, dsn: str):
        if not dsn.strip():
            raise ValueError("Postgres DSN is empty")
        self.dsn = dsn
        self._psycopg = None

    def _connect(self):
        if self._psycopg is None:
            try:
                import psycopg  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "PostgreSQL backend requires `psycopg` package. Install with `pip install psycopg[binary]`."
                ) from e
            self._psycopg = psycopg
        return self._psycopg.connect(self.dsn)

    def initialize(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_sentiment_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        captured_at TIMESTAMPTZ NOT NULL,
                        news_date TEXT,
                        total_articles INTEGER NOT NULL,
                        sentiment_avg DOUBLE PRECISION NOT NULL,
                        positive_count INTEGER NOT NULL,
                        negative_count INTEGER NOT NULL,
                        neutral_count INTEGER NOT NULL,
                        source TEXT,
                        payload_json JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_decisions (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        query_model TEXT NOT NULL,
                        resolved_model TEXT NOT NULL,
                        data_date TEXT NOT NULL,
                        selected_action TEXT NOT NULL,
                        raw_action TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        entropy DOUBLE PRECISION NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        safe_mode BOOLEAN NOT NULL,
                        safe_reason TEXT,
                        action_probs_json JSONB NOT NULL,
                        expected_rewards_json JSONB NOT NULL,
                        news_context_json JSONB,
                        decision_json JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at ON agent_decisions(created_at)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_decisions_model ON agent_decisions(resolved_model)"
                )
            conn.commit()

    def _normalize(self, decision: Any) -> Dict[str, Any]:
        if is_dataclass(decision):
            return asdict(decision)
        return dict(decision)

    def save_market_sentiment(self, snapshot: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO market_sentiment_snapshots (
                        captured_at, news_date, total_articles, sentiment_avg,
                        positive_count, negative_count, neutral_count, source, payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        json.dumps(snapshot, ensure_ascii=False),
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
        payload = self._normalize(decision)
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_decisions (
                        created_at, query_model, resolved_model, data_date,
                        selected_action, raw_action, confidence, entropy, value,
                        safe_mode, safe_reason, action_probs_json, expected_rewards_json,
                        news_context_json, decision_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        now,
                        query_model,
                        resolved_model,
                        data_date,
                        str(payload.get("action", "")),
                        str(payload.get("raw_action", "")),
                        float(payload.get("confidence", 0.0)),
                        float(payload.get("entropy", 0.0)),
                        float(payload.get("value", 0.0)),
                        bool(payload.get("safe_mode", False)),
                        payload.get("safe_reason"),
                        json.dumps(payload.get("action_probs", {}), ensure_ascii=False),
                        json.dumps(payload.get("expected_rewards", {}), ensure_ascii=False),
                        json.dumps(news_context or {}, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def health(self) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agent_decisions")
                decision_rows = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM market_sentiment_snapshots")
                sentiment_rows = int(cur.fetchone()[0])
        return {
            "backend": "postgres",
            "dsn_configured": True,
            "agent_decision_rows": decision_rows,
            "sentiment_rows": sentiment_rows,
        }
