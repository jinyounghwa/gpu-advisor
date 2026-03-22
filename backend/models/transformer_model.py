"""
0.1B Transformer 모델 - Policy-Value Dual Head
Mac M4 환경 최적화 (MPS 가속)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer with KV Cache"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_kv_cache: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_kv_cache = use_kv_cache

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, kv_cache: tuple = None
    ) -> tuple:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask
            kv_cache: (cached_keys, cached_values) for efficient inference

        Returns:
            output: (batch_size, seq_len, d_model)
            new_kv_cache: (new_keys, new_values) updated cache
        """
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_kv_cache and kv_cache is not None:
            cached_k, cached_v = kv_cache
            K = torch.cat([cached_k, K], dim=2)
            V = torch.cat([cached_v, V], dim=2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        new_kv_cache = (K, V) if self.use_kv_cache else None

        return self.W_o(output), new_kv_cache


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with KV Cache"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_kv_cache: bool = False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, use_kv_cache)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, kv_cache: tuple = None
    ) -> tuple:
        new_kv_cache = {}

        attn_output, layer_kv_cache = self.attention(x, mask, kv_cache)
        x = self.norm1(x + self.dropout(attn_output))

        if kv_cache is not None:
            new_kv_cache["attention"] = layer_kv_cache

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, new_kv_cache


class PolicyValueNetwork(nn.Module):
    """
    0.1B Transformer 모델 - Policy-Value Dual Head with KV Cache

    Architecture:
    - Input: 5개 변수 벡터
    - Shared: Transformer Blocks
    - Output:
        - Policy Head: Softmax (2 actions)
        - Value Head: Tanh (scalar value)
    """

    def __init__(
        self,
        input_dim: int = 11,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        num_actions: int = 2,
        dropout: float = 0.1,
        use_kv_cache: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_kv_cache = use_kv_cache

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout, use_kv_cache)
                for _ in range(num_layers)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> tuple:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            kv_cache: dict of cached KV pairs

        Returns:
            policy: (batch_size, num_actions)
            value: (batch_size, 1)
            new_kv_cache: dict of updated KV pairs
        """
        batch_size, seq_len, _ = x.size()

        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        new_kv_cache = {} if self.use_kv_cache else None

        for i, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get(f"layer_{i}") if kv_cache else None
            x, layer_new_cache = block(x, None, layer_cache)
            if self.use_kv_cache:
                new_kv_cache[f"layer_{i}"] = layer_new_cache

        last_token = x[:, -1, :]

        policy = self.policy_head(last_token)
        policy = F.softmax(policy, dim=-1)

        value = self.value_head(last_token)
        value = torch.tanh(value)

        return policy, value, new_kv_cache

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def create_model(config: dict) -> PolicyValueNetwork:
    """
    모델 생성 함수

    Args:
        config: 모델 설정 딕셔너리

    Returns:
        PolicyValueNetwork 인스턴스
    """
    model = PolicyValueNetwork(
        input_dim=config.get("input_dim", 11),
        d_model=config.get("d_model", 256),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        d_ff=config.get("d_ff", 1024),
        num_actions=config.get("num_actions", 2),
        dropout=config.get("dropout", 0.1),
    )

    # MPS 장치 설정 (Mac M4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"Model created with {model.count_parameters():,} parameters")
    print(f"Device: {device}")

    return model


if __name__ == "__main__":
    # 테스트
    config = {
        "input_dim": 11,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 1024,
        "num_actions": 2,
    }

    model = create_model(config)

    # 테스트 입력
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 11)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = x.to(device)

    policy, value = model(x)

    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
