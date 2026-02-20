"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Dict

from fastapi.testclient import TestClient

# Mock the agent before importing the server
mock_decision = MagicMock()
mock_decision.gpu_model = "RTX 4090"
mock_decision.action = "WAIT_SHORT"
mock_decision.raw_action = "WAIT_SHORT"
mock_decision.confidence = 0.65
mock_decision.entropy = 1.2
mock_decision.value = 0.3
mock_decision.simulations = 50
mock_decision.safe_mode = False
mock_decision.safe_reason = None
mock_decision.action_probs = {"BUY_NOW": 0.2, "WAIT_SHORT": 0.4, "WAIT_LONG": 0.2, "HOLD": 0.1, "SKIP": 0.1}
mock_decision.expected_rewards = {"BUY_NOW": -0.01, "WAIT_SHORT": 0.02, "WAIT_LONG": 0.01, "HOLD": 0.0, "SKIP": -0.005}
mock_decision.date = "2026-02-21"


@pytest.fixture
def client():
    """Create test client with mocked agent."""
    mock_agent = MagicMock()
    mock_agent.decide.return_value = mock_decision
    mock_agent.explain.return_value = "단기 대기가 더 유리하다고 판단했습니다."
    mock_agent.get_model_info.return_value = {
        "checkpoint_path": "test.pth",
        "device": "cpu",
        "num_simulations": 50,
        "min_confidence": 0.25,
        "max_entropy": 1.58,
        "meta": {},
    }

    with patch("backend.simple_server.get_gpu_agent", return_value=mock_agent):
        from backend.simple_server import app
        yield TestClient(app)


class TestHealthCheck:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestAskEndpoint:
    def test_valid_query(self, client):
        resp = client.post("/api/ask", json={"model_name": "RTX 4090"})
        assert resp.status_code == 200
        data = resp.json()
        assert "title" in data
        assert "summary" in data
        assert "agent_trace" in data

    def test_empty_model_name(self, client):
        resp = client.post("/api/ask", json={"model_name": ""})
        assert resp.status_code == 400

    def test_missing_field(self, client):
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422


class TestTrainingEndpoints:
    def test_training_status(self, client):
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data
        assert data["is_training"] is False

    def test_training_metrics_empty(self, client):
        resp = client.get("/api/training/metrics?last_n=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert data["total"] == 0

    def test_training_summary_empty(self, client):
        resp = client.get("/api/training/summary")
        assert resp.status_code == 200


class TestSystemStatus:
    def test_system_status(self, client):
        resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "memory_mb" in data
        assert "is_training" in data
