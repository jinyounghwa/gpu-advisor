"""
Prediction Network f(s) - Policy Value Network
AlphaGo의 Policy-Value Network (MCTS를 위해 수정)

입력:
    s_t: 현재 Latent State (batch_size, latent_dim=256)

출력:
    π(s_t): Policy - 행동 확률 분포 (batch_size, action_dim=5)
    v(s_t): Value - 상태 가치 (batch_size, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PredictionNetwork(nn.Module):
    """
    Policy-Value Network f(s)

    AlphaGo의 Policy-Value Network을 바탕으로 MCTS에 적용

    입력:
        s_t: 현재 Latent State (batch_size, latent_dim=256)

    출력:
        π(a|s): Policy logits (batch_size, action_dim=5)
        v(s): Value (batch_size, 1)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        action_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared trunk
        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # Shared transformer-style blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Separate heads
        # 1. Policy head (MCTS용) - 더 많은 탐색 가능하도록 크게
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, action_dim),
        )

        # 2. Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Value 범위: [-1, 1]
        )

    def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t: 현재 Latent State (batch_size, latent_dim=256)

        Returns:
            π(s_t): Policy logits (batch_size, 5)
            v(s_t): Value (batch_size, 1)
        """
        x = self.input_layer(s_t)
        x = self.layer_norm1(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm2(x)

        # Policy head
        policy_logits = self.policy_head(x)  # (batch_size, 5)

        # Value head
        value = self.value_head(x).squeeze(-1)  # (batch_size)

        return policy_logits, value


class TransitionModel(nn.Module):
    """
    Transition Model
    P(s_{t+1} | s_t, a_t) ∝ P(s_t+1)
    = P(s_t) (Gaussian with learned dynamics)
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Prior: P(s_t+1 | s_t, a_t)
        """
        return self.transition(s_t)


if __name__ == "__main__":
    # 테스트
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PredictionNetwork().to(device)

    # 샘플 입력
    batch_size = 4
    latent_dim = 256

    s_t = torch.randn(batch_size, latent_dim).to(device)
    policy_logits, value = model(s_t)

    print(f"Policy logits shape: {policy_logits.shape}")  # (4, 5)
    print(f"Value shape: {value.shape}")  # (4,)

    # Action probabilities
    action_probs = F.softmax(policy_logits, dim=-1)
    print(f"Action probabilities: {action_probs[0]}")  # (4, 5)
    print(f"Value: {value}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
