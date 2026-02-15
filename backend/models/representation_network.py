"""
Representation Network h(s) - 시장 State를 Latent State로 인코딩
AlphaGo/MuZero 스타일의 World Model 첫 번째 네트워크
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class FeedForward(nn.Module):
    """Feed Forward Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class RepresentationNetwork(nn.Module):
    """
    Representation Network h(s)
    시장 State(s_t) → Latent State(s_0)

    입력: 22차원 시장 State
        - OHLCV (5): price_1m, price_5m, price_15m, price_1h, price_change_24h
        - Order Book (3): imbalance_l1, imbalance_l2, imbalance_l3
        - Account (4): position, unrealized_pnl, time_in_trade
        - Macro Factors (4): fed_rate, korea_rate, usd_krw, usd_jpy
        - Time (1): timestamp

    출력: 256차원 Latent State(s_0)
    """

    def __init__(
        self,
        state_dim: int = 22,  # 22차원 시장 State
        latent_dim: int = 256,  # 256차원 Latent State
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1,
    ):
        super().__init__()

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len)

        # Input embedding
        self.input_embedding = nn.Linear(state_dim, latent_dim)
        self.layer_norm1 = nn.LayerNorm(latent_dim)

        # Feed Forward Networks (x3 for ensemble)
        self.ff1 = FeedForward(latent_dim, d_ff, dropout)
        self.ff2 = FeedForward(latent_dim, d_ff, dropout)
        self.ff3 = FeedForward(latent_dim, d_ff, dropout)

        self.layer_norm2 = nn.LayerNorm(latent_dim)

        # Output projection
        self.output_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_tensor: (batch_size, state_dim)

        Returns:
            s_0: (batch_size, latent_dim)
        """
        if state_tensor.dim() == 2:
            x = state_tensor.unsqueeze(1)  # (batch_size, 1, state_dim)
        else:
            x = state_tensor

        # Input embedding
        x = self.input_embedding(x)  # (batch_size, 1, latent_dim)
        x = self.layer_norm1(x)

        # Position encoding
        x = self.pos_encoding(x)

        # Ensemble of FeedForward networks
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.ff3(x)

        x = self.layer_norm2(x)

        # Output projection
        s_0 = self.output_layer(x)
        
        return s_0.squeeze(1)  # (batch_size, latent_dim)


if __name__ == "__main__":
    # 테스트
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state_dim = 22
    model = RepresentationNetwork(state_dim=state_dim).to(device)

    # 샘플 입력 (batch_size=4, state_dim=22)
    batch_size = 4
    dummy_input = torch.randn(batch_size, state_dim).to(device)

    print(f"Input shape: {dummy_input.shape}")

    s_0 = model(dummy_input)
    print(f"Latent State shape: {s_0.shape}")  # (4, 256)
    print(f"Latent State mean: {s_0.mean().item():.4f}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
