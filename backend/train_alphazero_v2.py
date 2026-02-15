"""
AlphaZero/MuZero Training Loop
Policy-Value Loss, Self-Play, MPS Acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from models.transformer_model import PolicyValueNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork
from models.mcts_engine import MCTSConfig, MCTSTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero Loss Function
    Combined Policy Loss (Cross-Entropy) and Value Loss (MSE)
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def forward(
        self,
        policy_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute AlphaZero loss

        Args:
            policy_pred: predicted policy (batch_size, num_actions)
            policy_target: target policy from MCTS (batch_size, num_actions)
            value_pred: predicted value (batch_size, 1)
            value_target: target value (batch_size, 1)

        Returns:
            dict of losses: {'total', 'policy', 'value', 'entropy'}
        """
        policy_loss = F.cross_entropy(policy_pred, policy_target)

        value_loss = F.mse_loss(value_pred.squeeze(-1), value_target.squeeze(-1))

        policy_entropy = -torch.sum(
            policy_pred * torch.log(policy_pred + 1e-10), dim=-1
        ).mean()

        total_loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            - self.entropy_weight * policy_entropy
        )

        return {
            "total": total_loss,
            "policy": policy_loss,
            "value": value_loss,
            "entropy": policy_entropy,
        }


class SelfPlayDataset(TensorDataset):
    """Self-play training dataset"""

    def __init__(self, episodes: List[Dict]):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class AlphaZeroTrainer:
    """
    AlphaZero/MuZero Training Loop
    """

    def __init__(
        self,
        policy_network: PolicyValueNetwork,
        dynamics_network: DynamicsNetwork,
        prediction_network: PredictionNetwork,
        config: Dict,
        device: str = "mps",
    ):
        self.policy_network = policy_network.to(device)
        self.dynamics_network = dynamics_network.to(device)
        self.prediction_network = prediction_network.to(device)

        self.device = device
        self.config = config

        self.loss_fn = AlphaZeroLoss(
            policy_weight=config.get("policy_weight", 1.0),
            value_weight=config.get("value_weight", 1.0),
            entropy_weight=config.get("entropy_weight", 0.01),
        )

        self.optimizer_policy = optim.AdamW(
            self.policy_network.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.optimizer_dynamics = optim.AdamW(
            self.dynamics_network.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.optimizer_prediction = optim.AdamW(
            self.prediction_network.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        mcts_config = MCTSConfig(
            num_simulations=config.get("mcts_simulations", 50),
            exploration=config.get("mcts_exploration", 1.4142),
            dirichlet_alpha=config.get("dirichlet_alpha", 0.03),
            rollout_steps=config.get("rollout_steps", 5),
        )

        self.mcts_trainer = MCTSTrainer(
            self.prediction_network, self.dynamics_network, mcts_config, device
        )

        self.scheduler_policy = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_policy,
            T_max=config.get("num_training_steps", 100000),
            eta_min=config.get("lr_min", 1e-6),
        )

        self.scheduler_dynamics = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dynamics,
            T_max=config.get("num_training_steps", 100000),
            eta_min=config.get("lr_min", 1e-6),
        )

        self.scheduler_prediction = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_prediction,
            T_max=config.get("num_training_steps", 100000),
            eta_min=config.get("lr_min", 1e-6),
        )

        logger.info("Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Policy network parameters: {sum(p.numel() for p in self.policy_network.parameters()):,}"
        )
        logger.info(
            f"Dynamics network parameters: {sum(p.numel() for p in self.dynamics_network.parameters()):,}"
        )
        logger.info(
            f"Prediction network parameters: {sum(p.numel() for p in self.prediction_network.parameters()):,}"
        )

    def generate_self_play_data(
        self, num_episodes: int, initial_states: np.ndarray
    ) -> List[Dict]:
        """
        Generate self-play training data using MCTS

        Args:
            num_episodes: number of episodes to generate
            initial_states: initial states for each episode

        Returns:
            list of training examples
        """
        all_training_data = []

        for i in range(num_episodes):
            episode = self.mcts_trainer.generate_episode(initial_states[i])
            all_training_data.extend(episode)

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_episodes} episodes")

        return all_training_data

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """
        Single training step

        Args:
            batch: list of training examples

        Returns:
            dict of losses and metrics
        """
        states = torch.stack(
            [torch.tensor(item["state"], dtype=torch.float32) for item in batch]
        ).to(self.device)

        actions = torch.tensor([item["action"] for item in batch], dtype=torch.long).to(
            self.device
        )

        target_policies = torch.stack(
            [torch.tensor(item["action_probs"], dtype=torch.float32) for item in batch]
        ).to(self.device)

        target_values = torch.tensor(
            [[item["value"]] for item in batch], dtype=torch.float32
        ).to(self.device)

        next_states = torch.stack(
            [torch.tensor(item["next_state"], dtype=torch.float32) for item in batch]
        ).to(self.device)

        rewards = torch.tensor(
            [[item["reward"]] for item in batch], dtype=torch.float32
        ).to(self.device)

        batch_size = states.shape[0]

        actions_onehot = F.one_hot(actions, num_classes=5).float().to(self.device)

        self.optimizer_policy.zero_grad()
        self.optimizer_dynamics.zero_grad()
        self.optimizer_prediction.zero_grad()

        policy_pred, value_pred, _ = self.policy_network(states)
        policy_loss = F.cross_entropy(policy_pred, actions)

        value_loss = F.mse_loss(value_pred.squeeze(-1), target_values.squeeze(-1))

        next_state_pred, reward_mean_pred, reward_logvar_pred = self.dynamics_network(
            states, actions_onehot
        )

        dynamics_state_loss = F.mse_loss(next_state_pred, next_states)
        dynamics_reward_loss = F.mse_loss(
            reward_mean_pred.squeeze(-1), rewards.squeeze(-1)
        )

        policy_logits_pred, value_pred_from_prediction = self.prediction_network(states)
        prediction_policy_loss = F.cross_entropy(policy_logits_pred, actions)
        prediction_value_loss = F.mse_loss(value_pred_from_prediction, target_values)

        policy_entropy = -torch.sum(
            policy_pred * torch.log(policy_pred + 1e-10), dim=-1
        ).mean()

        total_policy_loss = policy_loss + 0.5 * value_loss - 0.01 * policy_entropy

        total_dynamics_loss = dynamics_state_loss + dynamics_reward_loss
        total_prediction_loss = prediction_policy_loss + prediction_value_loss

        total_loss = total_policy_loss + total_dynamics_loss + total_prediction_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.dynamics_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self.prediction_network.parameters(), max_norm=1.0
        )

        self.optimizer_policy.step()
        self.optimizer_dynamics.step()
        self.optimizer_prediction.step()

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "dynamics_loss": total_dynamics_loss.item(),
            "prediction_loss": total_prediction_loss.item(),
            "entropy": policy_entropy.item(),
        }

    def train(
        self,
        num_training_steps: int,
        episodes_per_step: int,
        batch_size: int,
        initial_states: np.ndarray,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Main training loop

        Args:
            num_training_steps: number of training iterations
            episodes_per_step: number of self-play episodes per step
            batch_size: training batch size
            initial_states: initial states for episodes
            save_dir: directory to save checkpoints
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")
        logger.info(f"Training steps: {num_training_steps}")
        logger.info(f"Episodes per step: {episodes_per_step}")
        logger.info(f"Batch size: {batch_size}")

        for step in range(num_training_steps):
            sampled_indices = np.random.choice(
                len(initial_states), size=episodes_per_step, replace=True
            )
            sampled_states = initial_states[sampled_indices]

            training_data = self.generate_self_play_data(
                episodes_per_step, sampled_states
            )

            if len(training_data) < batch_size:
                continue

            indices = np.random.choice(
                len(training_data), size=batch_size, replace=True
            )
            batch = [training_data[i] for i in indices]

            metrics = self.train_step(batch)

            self.scheduler_policy.step()
            self.scheduler_dynamics.step()
            self.scheduler_prediction.step()

            if (step + 1) % 100 == 0:
                logger.info(f"Step {step + 1}/{num_training_steps}")
                logger.info(f"  Loss: {metrics['loss']:.4f}")
                logger.info(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                logger.info(f"  Value Loss: {metrics['value_loss']:.4f}")
                logger.info(f"  Dynamics Loss: {metrics['dynamics_loss']:.4f}")
                logger.info(f"  Prediction Loss: {metrics['prediction_loss']:.4f}")
                logger.info(f"  Entropy: {metrics['entropy']:.4f}")

                if save_dir:
                    self.save_checkpoint(save_dir, step + 1)

        logger.info("Training completed!")

    def save_checkpoint(self, save_dir: str, step: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "step": step,
            "policy_network": self.policy_network.state_dict(),
            "dynamics_network": self.dynamics_network.state_dict(),
            "prediction_network": self.prediction_network.state_dict(),
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_dynamics": self.optimizer_dynamics.state_dict(),
            "optimizer_prediction": self.optimizer_prediction.state_dict(),
            "scheduler_policy": self.scheduler_policy.state_dict(),
            "scheduler_dynamics": self.scheduler_dynamics.state_dict(),
            "scheduler_prediction": self.scheduler_prediction.state_dict(),
            "config": self.config,
        }

        path = Path(save_dir) / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    config = {
        "lr": 1e-4,
        "lr_min": 1e-6,
        "weight_decay": 1e-5,
        "mcts_simulations": 50,
        "mcts_exploration": 1.4142,
        "dirichlet_alpha": 0.03,
        "rollout_steps": 5,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "entropy_weight": 0.01,
    }

    latent_dim = 256
    action_dim = 5

    policy_network = PolicyValueNetwork(
        input_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        num_actions=action_dim,
        use_kv_cache=False,
    )

    dynamics_network = DynamicsNetwork(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=512, num_layers=4
    )

    prediction_network = PredictionNetwork(
        latent_dim=latent_dim, hidden_dim=512, num_layers=4, action_dim=action_dim
    )

    trainer = AlphaZeroTrainer(
        policy_network, dynamics_network, prediction_network, config, device
    )

    num_episodes = 100
    initial_states = np.random.randn(num_episodes, latent_dim)

    training_data = trainer.generate_self_play_data(10, initial_states[:10])

    logger.info(f"Generated {len(training_data)} training examples")

    batch = training_data[:4]
    metrics = trainer.train_step(batch)

    logger.info(f"Training step metrics: {metrics}")
