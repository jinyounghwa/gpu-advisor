"""Common repository protocol for persistence backends."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class RepositoryProtocol(Protocol):
    def initialize(self) -> None: ...

    def save_market_sentiment(self, snapshot: Dict[str, Any]) -> None: ...

    def save_agent_decision(
        self,
        *,
        query_model: str,
        resolved_model: str,
        data_date: str,
        decision: Any,
        news_context: Dict[str, Any] | None = None,
    ) -> None: ...

    def health(self) -> Dict[str, Any]: ...
