"""Tests for MCTS engine."""

import pytest
import torch
import numpy as np

from backend.models.mcts_engine import MCTSConfig, MCTSEngine, MCTSNode


class DummyPolicy:
    """Deterministic dummy policy for testing."""

    def __call__(self, state):
        batch = state.shape[0]
        logits = torch.zeros(batch, 5)
        logits[:, 0] = 2.0  # bias towards action 0
        value = torch.zeros(batch)
        return logits, value

    def to(self, device):
        return self


class DummyDynamics:
    """Deterministic dummy dynamics for testing."""

    def __call__(self, state, action):
        next_state = state + 0.01
        reward = torch.ones(state.shape[0])
        logvar = torch.zeros(state.shape[0])
        return next_state, reward, logvar

    def to(self, device):
        return self


class TestMCTSNode:
    def test_initial_value(self):
        node = MCTSNode(state=np.zeros(256))
        assert node.value == 0.0
        assert node.visits == 0
        assert not node.is_expanded

    def test_ucb_score_unvisited(self):
        node = MCTSNode(state=np.zeros(256))
        assert node.ucb_score == float("inf")

    def test_value_after_updates(self):
        node = MCTSNode(state=np.zeros(256))
        node.visits = 10
        node.total_value = 5.0
        assert node.value == 0.5


class TestMCTSEngine:
    def setup_method(self):
        self.config = MCTSConfig(num_simulations=10, rollout_steps=2)
        self.engine = MCTSEngine(self.config, latent_dim=256, action_dim=5)

    def test_search_returns_valid_probs(self):
        root_state = np.zeros(256)
        probs, value, tree = self.engine.search(
            root_state, DummyPolicy(), DummyDynamics(), device="cpu"
        )
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_root_is_expanded_after_search(self):
        root_state = np.zeros(256)
        _, _, tree = self.engine.search(
            root_state, DummyPolicy(), DummyDynamics(), device="cpu"
        )
        assert tree.is_expanded
        assert tree.visits > 0
        assert len(tree.children) == 5

    def test_search_with_nonzero_state(self):
        root_state = np.random.randn(256).astype(np.float32)
        probs, value, tree = self.engine.search(
            root_state, DummyPolicy(), DummyDynamics(), device="cpu"
        )
        assert probs.shape == (5,)
        assert tree.visits == self.config.num_simulations

    def test_temperature_zero_greedy(self):
        """Temperature 0 should select a single action deterministically."""
        root_state = np.zeros(256)
        config = MCTSConfig(num_simulations=20, rollout_steps=2, temperature=0)
        engine = MCTSEngine(config, latent_dim=256, action_dim=5)
        probs, _, _ = engine.search(
            root_state, DummyPolicy(), DummyDynamics(), device="cpu"
        )
        assert max(probs) == 1.0
        assert sum(p == 0 for p in probs) == 4
