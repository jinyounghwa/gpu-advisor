"""
Inference Engine for AlphaZero/MuZero Models
Lightweight service for real-time predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time

from models.transformer_model import PolicyValueNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork
from models.mcts_engine import MCTSConfig, MCTSEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    AlphaZero Inference Engine
    Real-time prediction service
    """

    def __init__(
        self,
        policy_network: PolicyValueNetwork,
        dynamics_network: Optional[DynamicsNetwork] = None,
        prediction_network: Optional[PredictionNetwork] = None,
        config: Dict = None,
        use_mcts: bool = True,
        device: str = "mps",
    ):
        self.policy_network = policy_network.to(device)
        self.dynamics_network = (
            dynamics_network.to(device) if dynamics_network else None
        )
        self.prediction_network = (
            prediction_network.to(device) if prediction_network else None
        )

        self.device = device
        self.use_mcts = use_mcts

        if config:
            self.mcts_config = MCTSConfig(
                num_simulations=config.get("mcts_simulations", 50),
                exploration=config.get("mcts_exploration", 1.4142),
                dirichlet_alpha=config.get("dirichlet_alpha", 0.03),
                rollout_steps=config.get("rollout_steps", 5),
            )
        else:
            self.mcts_config = MCTSConfig()

        if use_mcts and prediction_network and dynamics_network:
            self.mcts = MCTSEngine(self.mcts_config)
        else:
            self.mcts = None

        self.policy_network.eval()
        if self.dynamics_network:
            self.dynamics_network.eval()
        if self.prediction_network:
            self.prediction_network.eval()

        self.kv_cache = None

        logger.info("Inference engine initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Use MCTS: {self.use_mcts}")

    def predict(
        self, state: np.ndarray, use_kv_cache: bool = False, temperature: float = 1.0
    ) -> Dict:
        """
        Make prediction for given state

        Args:
            state: current state (latent_dim,)
            use_kv_cache: whether to use KV cache
            temperature: sampling temperature

        Returns:
            dict with predictions
        """
        start_time = time.time()

        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            if (
                self.use_mcts
                and self.mcts
                and self.dynamics_network
                and self.prediction_network
            ):
                action_probs, value, tree = self.mcts.search(
                    state, self.prediction_network, self.dynamics_network
                )

                if temperature > 0:
                    action_probs = action_probs ** (1.0 / temperature)
                    action_probs = action_probs / action_probs.sum()
                else:
                    best_action = np.argmax(action_probs)
                    action_probs = np.zeros_like(action_probs)
                    action_probs[best_action] = 1.0

                action = np.random.choice(len(action_probs), p=action_probs)

            else:
                policy, value, new_kv_cache = self.policy_network(
                    state_tensor, kv_cache=self.kv_cache if use_kv_cache else None
                )

                if use_kv_cache:
                    self.kv_cache = new_kv_cache

                policy_probs = policy.cpu().numpy()[0]
                value = value.cpu().numpy()[0]

                if temperature > 0:
                    action_probs = policy_probs ** (1.0 / temperature)
                    action_probs = action_probs / action_probs.sum()
                else:
                    best_action = np.argmax(action_probs)
                    action_probs = np.zeros_like(action_probs)
                    action_probs[best_action] = 1.0

                action = np.random.choice(len(action_probs), p=action_probs)

        inference_time = time.time() - start_time

        return {
            "action": int(action),
            "action_probs": action_probs,
            "value": float(value),
            "inference_time_ms": inference_time * 1000,
            "tree_depth": tree.visits
            if self.use_mcts and hasattr(tree, "visits")
            else 0,
        }

    def predict_batch(
        self, states: np.ndarray, use_kv_cache: bool = False
    ) -> List[Dict]:
        """
        Batch prediction

        Args:
            states: batch of states (batch_size, latent_dim)
            use_kv_cache: whether to use KV cache

        Returns:
            list of predictions
        """
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            policies, values, _ = self.policy_network(
                states_tensor, kv_cache=self.kv_cache if use_kv_cache else None
            )

        if use_kv_cache:
            self.kv_cache = _

        policy_probs = torch.softmax(policies, dim=-1).cpu().numpy()
        values = values.cpu().numpy()

        predictions = []
        for i in range(len(states)):
            predictions.append(
                {
                    "action": int(np.argmax(policy_probs[i])),
                    "action_probs": policy_probs[i],
                    "value": float(values[i]),
                    "inference_time_ms": 0,
                    "tree_depth": 0,
                }
            )

        return predictions

    def simulate_next_state(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """
        Simulate next state using dynamics model

        Args:
            state: current state
            action: action to take

        Returns:
            next_state, reward
        """
        if not self.dynamics_network:
            raise ValueError("Dynamics network not available")

        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        action_onehot = np.zeros(self.mcts_config.dirichlet_alpha + 2)
        action_onehot[action] = 1.0
        action_tensor = (
            torch.tensor(action_onehot, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            next_state, reward, _ = self.dynamics_network(state_tensor, action_tensor)

        next_state_np = next_state.squeeze(0).cpu().numpy()
        reward_np = reward.squeeze(0).item()

        return next_state_np, reward_np

    def benchmark(
        self, num_samples: int = 1000, use_kv_cache: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark inference speed

        Args:
            num_samples: number of samples for benchmark
            use_kv_cache: whether to use KV cache

        Returns:
            benchmark metrics
        """
        logger.info(f"Running benchmark with {num_samples} samples...")
        logger.info(f"Use KV Cache: {use_kv_cache}")

        test_states = np.random.randn(num_samples, 256).astype(np.float32)

        latencies = []

        for state in test_states:
            start = time.time()
            _ = self.predict(state, use_kv_cache=use_kv_cache)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        metrics = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_samples_per_sec": num_samples / sum(latencies) * 1000,
            "total_samples": num_samples,
        }

        logger.info("\nBenchmark Results:")
        logger.info(f"  Mean Latency: {metrics['mean_latency_ms']:.2f} ms")
        logger.info(f"  Median Latency: {metrics['median_latency_ms']:.2f} ms")
        logger.info(f"  P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
        logger.info(f"  P99 Latency: {metrics['p99_latency_ms']:.2f} ms")
        logger.info(
            f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec"
        )

        return metrics


def load_inference_engine(
    checkpoint_path: str,
    device: str = "mps",
    use_mcts: bool = True,
    use_quantized: bool = False,
) -> InferenceEngine:
    """
    Load inference engine from checkpoint

    Args:
        checkpoint_path: path to model checkpoint
        device: device to use
        use_mcts: whether to use MCTS
        use_quantized: whether to load quantized model

    Returns:
        InferenceEngine instance
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    latent_dim = 256
    action_dim = 5

    policy_network = PolicyValueNetwork(
        input_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        num_actions=action_dim,
        use_kv_cache=True,
    )

    dynamics_network = DynamicsNetwork(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=512, num_layers=4
    )

    prediction_network = PredictionNetwork(
        latent_dim=latent_dim, hidden_dim=512, num_layers=4, action_dim=action_dim
    )

    if "policy_network" in checkpoint:
        policy_network.load_state_dict(checkpoint["policy_network"])
    if "dynamics_network" in checkpoint:
        dynamics_network.load_state_dict(checkpoint["dynamics_network"])
    if "prediction_network" in checkpoint:
        prediction_network.load_state_dict(checkpoint["prediction_network"])

    config = checkpoint.get("config", {})

    engine = InferenceEngine(
        policy_network,
        dynamics_network,
        prediction_network,
        config,
        use_mcts=use_mcts,
        device=device,
    )

    logger.info("Inference engine loaded successfully")

    return engine


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    latent_dim = 256
    action_dim = 5

    policy_network = PolicyValueNetwork(
        input_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        num_actions=action_dim,
        use_kv_cache=True,
    )

    dummy_dynamics = DynamicsNetwork(latent_dim=latent_dim, action_dim=action_dim)
    dummy_prediction = PredictionNetwork(latent_dim=latent_dim, action_dim=action_dim)

    config = {
        "mcts_simulations": 50,
        "mcts_exploration": 1.4142,
        "dirichlet_alpha": 0.03,
        "rollout_steps": 5,
    }

    engine = InferenceEngine(
        policy_network,
        dummy_dynamics,
        dummy_prediction,
        config,
        use_mcts=True,
        device=device,
    )

    test_state = np.random.randn(latent_dim).astype(np.float32)

    prediction = engine.predict(test_state)

    logger.info("\nSingle prediction test:")
    logger.info(f"  Action: {prediction['action']}")
    logger.info(f"  Action probs: {prediction['action_probs']}")
    logger.info(f"  Value: {prediction['value']:.4f}")
    logger.info(f"  Inference time: {prediction['inference_time_ms']:.2f} ms")

    metrics = engine.benchmark(num_samples=100)

    logger.info(f"\nBenchmark complete")
