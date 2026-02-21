"""E2E test: crawler -> feature engineering -> API inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from crawlers.danawa_crawler import DanawaCrawler
from crawlers.exchange_rate_crawler import ExchangeRateCrawler
from crawlers.feature_engineer import FeatureEngineer
from crawlers.news_crawler import NewsCrawler


@dataclass
class DummyDecision:
    gpu_model: str
    action: str
    raw_action: str
    confidence: float
    entropy: float
    value: float
    action_probs: dict[str, float]
    expected_rewards: dict[str, float]
    date: str
    simulations: int
    safe_mode: bool
    safe_reason: str | None


class DummyAgent:
    def __init__(self, dataset_path: Path):
        import json

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.rows = json.load(f)

    def resolve_state(self, query_model: str):
        for row in self.rows:
            if row["gpu_model"].lower() == query_model.lower():
                return row["gpu_model"], np.asarray(row["state_vector"], dtype=np.float32), row["date"]
        raise ValueError(f"unknown model: {query_model}")

    def decide_from_state(self, gpu_model: str, state_vec: np.ndarray, data_date: str) -> DummyDecision:
        sentiment_signal = float(state_vec[80]) if len(state_vec) > 80 else 0.5
        action = "WAIT_SHORT" if sentiment_signal >= 0.5 else "HOLD"
        return DummyDecision(
            gpu_model=gpu_model,
            action=action,
            raw_action=action,
            confidence=0.65,
            entropy=1.1,
            value=0.2,
            action_probs={"BUY_NOW": 0.2, "WAIT_SHORT": 0.45, "WAIT_LONG": 0.2, "HOLD": 0.1, "SKIP": 0.05},
            expected_rewards={"BUY_NOW": -0.01, "WAIT_SHORT": 0.02, "WAIT_LONG": 0.01, "HOLD": 0.0, "SKIP": -0.01},
            date=data_date,
            simulations=50,
            safe_mode=False,
            safe_reason=None,
        )

    def explain(self, decision: DummyDecision) -> str:
        return f"{decision.action} 선택"

    def decide(self, query_model: str):
        gpu_model, state_vec, data_date = self.resolve_state(query_model)
        return self.decide_from_state(gpu_model, state_vec, data_date)

    def get_model_info(self):
        return {"checkpoint_path": "dummy", "device": "cpu", "num_simulations": 50}


def test_e2e_crawler_to_api(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        DanawaCrawler,
        "crawl_all",
        lambda self: [
            {
                "product_name": "MSI GeForce RTX 4090",
                "manufacturer": "MSI",
                "chipset": "RTX 4090",
                "lowest_price": 2190000,
                "seller_count": 7,
                "stock_status": "in_stock",
                "product_url": "https://example.com/rtx4090",
            }
        ],
    )
    monkeypatch.setattr(
        ExchangeRateCrawler,
        "fetch_exchange_rates",
        lambda self: {"USD/KRW": 1420.1, "JPY/KRW": 950.2, "EUR/KRW": 1538.4},
    )
    monkeypatch.setattr(
        NewsCrawler,
        "crawl_all",
        lambda self: [
            {
                "title": "GPU price drop expected",
                "url": "https://example.com/news1",
                "source": "Example",
                "published_at": "Sat, 21 Feb 2026 09:00:00 GMT",
                "sentiment": "positive",
                "sentiment_score": 0.8,
                "keywords": ["GPU price"],
            }
        ],
    )

    raw_root = tmp_path / "data" / "raw"
    processed_root = tmp_path / "data" / "processed"

    danawa = DanawaCrawler(output_dir=str(raw_root / "danawa"))
    danawa.target_gpus = ["RTX 4090"]
    products = danawa.crawl_all()
    danawa_file = Path(danawa.save(products))
    date_str = danawa_file.stem

    exchange = ExchangeRateCrawler(output_dir=str(raw_root / "exchange"))
    exchange.save(exchange.fetch_exchange_rates())

    news = NewsCrawler(output_dir=str(raw_root / "news"))
    news.keywords = ["GPU price"]
    news.save(news.crawl_all())

    engineer = FeatureEngineer(raw_data_dir=str(raw_root), processed_dir=str(processed_root))
    dataset_path = Path(engineer.process_all(date_str=date_str))
    assert dataset_path.exists()

    import backend.simple_server as server

    dummy_agent = DummyAgent(dataset_path)
    monkeypatch.setenv("GPU_ADVISOR_DB_PATH", str(processed_root / "test.db"))
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "get_gpu_agent", lambda: dummy_agent)
    server.repository = None

    client = TestClient(server.app)
    resp = client.post("/api/ask", json={"model_name": "RTX 4090"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["title"] == "RTX 4090"
    assert payload["agent_trace"]["news_context"]["applied"] is True

    health = server.get_repository().health()
    assert health["agent_decision_rows"] >= 1
    assert health["sentiment_rows"] >= 1
