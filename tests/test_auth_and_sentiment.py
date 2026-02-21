"""Auth flow and sentiment backend tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _mock_agent():
    decision = MagicMock()
    decision.gpu_model = "RTX 4090"
    decision.action = "WAIT_SHORT"
    decision.raw_action = "WAIT_SHORT"
    decision.confidence = 0.7
    decision.entropy = 1.1
    decision.value = 0.2
    decision.simulations = 50
    decision.safe_mode = False
    decision.safe_reason = None
    decision.action_probs = {"BUY_NOW": 0.2, "WAIT_SHORT": 0.5, "WAIT_LONG": 0.2, "HOLD": 0.05, "SKIP": 0.05}
    decision.expected_rewards = {"BUY_NOW": -0.01, "WAIT_SHORT": 0.02, "WAIT_LONG": 0.01, "HOLD": 0.0, "SKIP": -0.01}
    decision.date = "2026-02-21"

    agent = MagicMock()
    agent.decide.return_value = decision
    agent.explain.return_value = "단기 대기"
    return agent


def test_jwt_token_auth_flow(monkeypatch):
    import backend.simple_server as server

    monkeypatch.setattr(server, "get_gpu_agent", lambda: _mock_agent())
    monkeypatch.setattr(server.security_config, "auth_mode", server.AuthMode.JWT)
    monkeypatch.setattr(server.security_config, "users", {"tester": "secret"})

    client = TestClient(server.app)

    token_resp = client.post(
        "/api/auth/token",
        data={"username": "tester", "password": "secret"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    denied = client.post("/api/ask", json={"model_name": "RTX 4090"})
    assert denied.status_code == 401

    ok = client.post(
        "/api/ask",
        json={"model_name": "RTX 4090"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200


def test_sentiment_rule_backend(monkeypatch):
    monkeypatch.setenv("GPU_ADVISOR_SENTIMENT_BACKEND", "rule")
    from backend.api.sentiment.analyzer import NewsSentimentAnalyzer

    analyzer = NewsSentimentAnalyzer()
    result = analyzer.aggregate(
        [
            {"title": "GPU price drop expected soon"},
            {"title": "GPU shortage and volatility concerns"},
        ]
    )
    assert result["backend_used"] == "rule"
    assert "sentiment_avg" in result
