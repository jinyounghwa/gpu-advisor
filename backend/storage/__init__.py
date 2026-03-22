"""Persistence layer for production-friendly data storage."""

from __future__ import annotations

from pathlib import Path

from .base import RepositoryProtocol
from .postgres_repository import PostgresRepository
from .repository import SQLiteRepository


def create_repository(*, backend: str, sqlite_path: Path, postgres_dsn: str | None = None) -> RepositoryProtocol:
    backend_normalized = (backend or "sqlite").strip().lower()
    if backend_normalized == "postgres":
        return PostgresRepository(dsn=postgres_dsn or "")
    return SQLiteRepository(db_path=sqlite_path)


__all__ = ["RepositoryProtocol", "SQLiteRepository", "PostgresRepository", "create_repository"]
