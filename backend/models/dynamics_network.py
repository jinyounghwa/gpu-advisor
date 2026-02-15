"""
Dynamics Network g(s) - 시장 물리학 모델링
시장 State(s_t) + Action(a_t) → 다음 State(s_{t+1}) 추론
AlphaGo의 "World Model" 중 물리학 부분
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicsNetwork(nn.Module):
    """
    Dynamics Network g(s)
    (s_t, a_t) → (s_{t+1} + reward distribution)

    입력:
        s_t: 현재 Latent State (batch_size, latent_dim=256)
        a_t: Action (batch_size, action_dim=5) - BUY_SMALL, BUY_FULL, SELL_SMALL, SELL_FULL, HOLD

    출력:
        s_{t+1}: 다음 Latent State (batch_size, latent_dim=256)
        reward_dist: 미래 Reward 분포 (batch_size, 2) - [μ, σ²]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 5,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Concatenated input: [s_t; a_t]
        input_dim = latent_dim + action_dim

        # Input processing
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # Transformer-style blocks (but simpler for efficiency)
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

        # Output heads
        # 1. Next state prediction
        self.next_state_head = nn.Linear(hidden_dim, latent_dim)

        # 2. Reward distribution (Gaussian: μ and σ²)
        self.reward_mean_head = nn.Linear(hidden_dim, 1)
        self.reward_logvar_head = nn.Linear(
            hidden_dim, 1
        )  # log(σ²) for positive variance

    def forward(
        self, s_t: torch.Tensor, a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t: 현재 Latent State (batch_size, latent_dim=256)
            a_t: Action (batch_size, action_dim=5)

        Returns:
            s_{t+1}: 다음 Latent State (batch_size, latent_dim)
            reward_mean: Reward 분포 평균 μ (batch_size, 1)
            reward_logvar: Reward 분포 로그 분산 log(σ²) (batch_size, 1)
        """
        batch_size = s_t.size(0)

        # Concatenate state and action
        x = torch.cat([s_t, a_t], dim=-1)  # (batch_size, latent_dim + 5)

        # Input processing
        x = self.input_layer(x)
        x = self.layer_norm1(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.layer_norm2(x)

        # Next state prediction
        s_tp1 = self.next_state_head(x)  # (batch_size, latent_dim)

        # Reward distribution
        reward_mean = self.reward_mean_head(x).squeeze(-1)  # (batch_size)
        reward_logvar = self.reward_logvar_head(x).squeeze(-1)  # (batch_size, 1)

        # Softplus for log variance stability
        reward_logvar = F.softplus(reward_logvar, beta=1.0)

        return s_tp1, reward_mean, reward_logvar


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
    model = DynamicsNetwork().to(device)

    # 샘플 입력
    batch_size = 4
    latent_dim = 256
    action_dim = 5

    s_t = torch.randn(batch_size, latent_dim).to(device)
    a_t = torch.zeros(batch_size, action_dim).to(device)

    s_tp1, reward_mean, reward_logvar = model(s_t, a_t)
    reward_std = torch.exp(reward_logvar)  # σ

    print(f"s_t shape: {s_t.shape}")  # (4, 256)
    print(f"s_{t + 1} shape: {s_tp1.shape}")  # (4, 256)
    print(f"Reward mean: {reward_mean}")  # (4,)
    print(f"Reward std: {reward_std}")  # (4,)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
