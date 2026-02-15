"""
MCTS (Monte Carlo Tree Search) Engine for AlphaZero
Market trading scenario simulation with tree search
"""

import numpy as np
import math
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MCTSConfig:
    """MCTS hyperparameters"""

    num_simulations: int = 50
    exploration: float = 1.4142
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    rollout_steps: int = 5
    temperature: float = 1.0
    discount_factor: float = 0.99


@dataclass
class MCTSNode:
    """MCTS Tree Node"""

    state: np.ndarray
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0
    action: Optional[int] = None
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    is_expanded: bool = False
    is_terminal: bool = False

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    @property
    def ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value + self.prior * math.sqrt(
            math.log(self.parent.visits + 1) / (1 + self.visits)
        )


class MCTSEngine:
    """
    MCTS Engine with AlphaZero-style search
    """

    def __init__(self, config: MCTSConfig, latent_dim: int = 256, action_dim: int = 5):
        self.config = config
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def search(
        self,
        root_state: np.ndarray,
        policy_network,
        dynamics_network,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, float, MCTSNode]:
        """
        MCTS search

        Args:
            root_state: root state
            policy_network: prediction network f(s) -> (policy, value)
            dynamics_network: dynamics network g(s, a) -> (s', r)
            device: device for tensor placement

        Returns:
            action_probs: action probability distribution
            value: root value
            tree: search tree
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.config.num_simulations):
            node = self._select(root)

            if node.is_terminal:
                value = 0.0
            else:
                if not node.is_expanded:
                    node = self._expand(node, policy_network, device)

                value = self._simulate(node, policy_network, dynamics_network, device)

            self._backup(node, value)

        action_probs = self._get_action_probs(root, self.config.temperature)

        return action_probs, root.value, root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node for expansion using UCB"""
        while node.is_expanded and not node.is_terminal and node.children:
            node = max(node.children, key=lambda n: n.ucb_score)
        return node

    def _expand(self, node: MCTSNode, policy_network, device: str = "cpu") -> MCTSNode:
        """Expand node by adding children"""
        state_tensor = (
            torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            policy_logits, value = policy_network(state_tensor)

        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        if len(policy_probs.shape) == 0:
            policy_probs = np.ones(self.action_dim) / self.action_dim
        elif policy_probs.ndim > 1:
            policy_probs = policy_probs.mean(axis=0)

        if len(policy_probs) < self.action_dim:
            policy_probs = np.concatenate(
                [
                    policy_probs,
                    np.ones(self.action_dim - len(policy_probs)) / self.action_dim,
                ]
            )

        policy_probs = policy_probs / policy_probs.sum()

        node.is_expanded = True
        node.children = []

        for action in range(self.action_dim):
            child_state = node.state.copy()
            child = MCTSNode(
                state=child_state,
                prior=policy_probs[action],
                action=action,
                parent=node,
            )
            node.children.append(child)

        best_child = max(node.children, key=lambda n: n.prior)
        return best_child

    def _simulate(
        self, node: MCTSNode, policy_network, dynamics_network, device: str = "cpu"
    ) -> float:
        """Simulate trajectory using dynamics model"""
        total_reward = 0.0
        current_node = node
        discount = 1.0

        for step in range(self.config.rollout_steps):
            state_tensor = (
                torch.tensor(current_node.state, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                policy_logits, value = policy_network(state_tensor)
                policy_probs = torch.softmax(policy_logits, dim=-1)

            action = torch.multinomial(policy_probs, num_samples=1).item()

            action_onehot = np.zeros(self.action_dim)
            action_onehot[action] = 1.0

            state_tensor = (
                torch.tensor(current_node.state, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            action_tensor = (
                torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                next_state, reward_mean, _ = dynamics_network(
                    state_tensor, action_tensor
                )

            next_state_np = next_state.squeeze(0).cpu().numpy()

            reward = reward_mean.squeeze(0).cpu().item()

            total_reward += discount * reward
            discount *= self.config.discount_factor

            current_node = MCTSNode(state=next_state_np)

        final_state_tensor = (
            torch.tensor(current_node.state, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            _, value = policy_network(final_state_tensor)

        return total_reward + discount * value.item()

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Backup value through path"""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _get_action_probs(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """Get action probability distribution"""
        visits = np.array([child.visits for child in root.children])

        if temperature == 0:
            action_probs = np.zeros_like(visits, dtype=float)
            action_probs[np.argmax(visits)] = 1.0
        else:
            action_probs = visits ** (1.0 / temperature)
            action_probs = action_probs / action_probs.sum()

        if len(action_probs) < self.action_dim:
            padded = np.zeros(self.action_dim)
            padded[: len(action_probs)] = action_probs
            action_probs = padded

        return action_probs


class MCTSTrainer:
    """MCTS-based training with self-play"""

    def __init__(
        self, policy_network, dynamics_network, config: MCTSConfig, device: str = "mps"
    ):
        self.policy_network = policy_network.to(device)
        self.dynamics_network = dynamics_network.to(device)
        self.config = config
        self.device = device
        self.action_dim = 5
        self.mcts = MCTSEngine(config)

    def generate_episode(self, initial_state: np.ndarray) -> List[Dict]:
        """Generate training episode using MCTS"""
        episode_data = []
        current_state = initial_state
        done = False
        step = 0

        while not done and step < 200:
            action_probs, value, tree = self.mcts.search(
                current_state, self.policy_network, self.dynamics_network, self.device
            )

            action = np.random.choice(len(action_probs), p=action_probs)

            action_onehot = np.zeros(self.action_dim)
            action_onehot[action] = 1.0

            state_tensor = (
                torch.tensor(current_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            action_tensor = (
                torch.tensor(action_onehot, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                next_state, reward, _ = self.dynamics_network(
                    state_tensor, action_tensor
                )

            next_state_np = next_state.squeeze(0).cpu().numpy()
            reward_val = reward.squeeze(0).item()

            episode_data.append(
                {
                    "state": current_state,
                    "action": action,
                    "action_probs": action_probs,
                    "reward": reward_val,
                    "value": value,
                    "next_state": next_state_np,
                }
            )

            current_state = next_state_np
            step += 1

            if step >= 200:
                done = True

        return episode_data


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class DummyPolicy:
        def __init__(self):
            self.latent_dim = 256

        def to(self, device):
            return self

        def __call__(self, state):
            return torch.randn(1, 5).to(device), torch.randn(1).to(device)

    class DummyDynamics:
        def __init__(self):
            self.latent_dim = 256

        def to(self, device):
            return self

        def __call__(self, state, action):
            batch_size = state.shape[0]
            next_state = state + torch.randn_like(state) * 0.01
            reward = torch.randn(batch_size, 1).to(device)
            logvar = torch.randn(batch_size, 1).to(device)
            return next_state, reward, logvar

    config = MCTSConfig(
        num_simulations=50, exploration=1.4142, dirichlet_alpha=0.03, rollout_steps=5
    )

    mcts = MCTSEngine(config=config, latent_dim=256, action_dim=5)

    root_state = np.zeros(256)
    action_probs, value, tree = mcts.search(root_state, DummyPolicy(), DummyDynamics())

    print(f"Action probs: {action_probs}")
    print(f"Value: {value}")
    print(f"Root visits: {tree.visits}")
