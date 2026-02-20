"""Tests for neural network architectures (Representation, Dynamics, Prediction)."""

import pytest
import torch
import numpy as np

from backend.models.representation_network import RepresentationNetwork
from backend.models.dynamics_network import DynamicsNetwork
from backend.models.prediction_network import PredictionNetwork


DEVICE = "cpu"
BATCH_SIZE = 4
STATE_DIM = 22
LATENT_DIM = 256
ACTION_DIM = 5


class TestRepresentationNetwork:
    def setup_method(self):
        self.model = RepresentationNetwork(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(DEVICE)

    def test_output_shape(self):
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        out = self.model(x)
        assert out.shape == (BATCH_SIZE, LATENT_DIM)

    def test_single_sample(self):
        x = torch.randn(1, STATE_DIM)
        out = self.model(x)
        assert out.shape == (1, LATENT_DIM)

    def test_deterministic(self):
        self.model.eval()
        x = torch.randn(1, STATE_DIM)
        with torch.no_grad():
            o1 = self.model(x)
            o2 = self.model(x)
        assert torch.allclose(o1, o2)

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 0
        # Should be in the millions range for 22->256 dim
        assert total > 100_000


class TestDynamicsNetwork:
    def setup_method(self):
        self.model = DynamicsNetwork(
            latent_dim=LATENT_DIM, action_dim=ACTION_DIM
        ).to(DEVICE)

    def test_output_shapes(self):
        s = torch.randn(BATCH_SIZE, LATENT_DIM)
        a = torch.zeros(BATCH_SIZE, ACTION_DIM)
        a[:, 0] = 1.0  # one-hot
        s_next, r_mean, r_logvar = self.model(s, a)
        assert s_next.shape == (BATCH_SIZE, LATENT_DIM)
        assert r_mean.shape == (BATCH_SIZE,)
        assert r_logvar.shape == (BATCH_SIZE,)

    def test_reward_logvar_positive(self):
        """Softplus should keep log-variance non-negative."""
        s = torch.randn(BATCH_SIZE, LATENT_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        _, _, r_logvar = self.model(s, a)
        assert (r_logvar >= 0).all()

    def test_different_actions_different_outputs(self):
        self.model.eval()
        s = torch.randn(1, LATENT_DIM)
        a1 = torch.zeros(1, ACTION_DIM)
        a1[0, 0] = 1.0
        a2 = torch.zeros(1, ACTION_DIM)
        a2[0, 4] = 1.0
        with torch.no_grad():
            s1, _, _ = self.model(s, a1)
            s2, _, _ = self.model(s, a2)
        # Different actions should (almost certainly) produce different next states
        assert not torch.allclose(s1, s2, atol=1e-6)


class TestPredictionNetwork:
    def setup_method(self):
        self.model = PredictionNetwork(
            latent_dim=LATENT_DIM, action_dim=ACTION_DIM
        ).to(DEVICE)

    def test_output_shapes(self):
        s = torch.randn(BATCH_SIZE, LATENT_DIM)
        policy, value = self.model(s)
        assert policy.shape == (BATCH_SIZE, ACTION_DIM)
        assert value.shape == (BATCH_SIZE,)

    def test_value_bounded(self):
        """Value head uses Tanh, output should be in [-1, 1]."""
        s = torch.randn(BATCH_SIZE, LATENT_DIM)
        _, value = self.model(s)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_policy_softmax_sums_to_one(self):
        s = torch.randn(BATCH_SIZE, LATENT_DIM)
        policy_logits, _ = self.model(s)
        probs = torch.softmax(policy_logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)


class TestEndToEndForward:
    """Test the full forward pass: state -> representation -> dynamics -> prediction."""

    def test_pipeline(self):
        h = RepresentationNetwork(state_dim=STATE_DIM, latent_dim=LATENT_DIM)
        g = DynamicsNetwork(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
        f = PredictionNetwork(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)

        h.eval()
        g.eval()
        f.eval()

        state = torch.randn(1, STATE_DIM)

        with torch.no_grad():
            latent = h(state)
            assert latent.shape == (1, LATENT_DIM)

            policy, value = f(latent)
            assert policy.shape == (1, ACTION_DIM)

            action = torch.zeros(1, ACTION_DIM)
            action[0, int(torch.argmax(policy))] = 1.0
            next_latent, reward_mean, _ = g(latent, action)
            assert next_latent.shape == (1, LATENT_DIM)

            # Can feed next_latent back into prediction
            policy2, value2 = f(next_latent)
            assert policy2.shape == (1, ACTION_DIM)
