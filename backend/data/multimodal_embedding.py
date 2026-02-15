"""
Multi-modal Embedding Layer
텍스트 토큰과 수치 데이터를 하나의 벡터 공간으로 융합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class MultiModalEmbedding(nn.Module):
    """
    멀티모달 임베딩 레이어

    특징:
    1. 텍스트 임베딩 (Learned Word Embeddings)
    2. 수치 임베딩 (Linear Projection)
    3. 포지셔널 인코딩
    4. 멀티모달 퓨전 (Concat + Linear)
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        text_embed_dim: int = 128,
        numerical_dim: int = 5,
        hidden_dim: int = 256,
        max_seq_length: int = 1000,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: 텍스트 어휘 크기
            text_embed_dim: 텍스트 임베딩 차원
            numerical_dim: 수치 데이터 차원
            hidden_dim: 공유 은닉 차원
            max_seq_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.text_embed_dim = text_embed_dim
        self.numerical_dim = numerical_dim
        self.hidden_dim = hidden_dim

        # 텍스트 임베딩
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.text_embedding.weight)

        # 수치 데이터 임베딩
        self.numerical_projection = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 멀티모달 퓨전 (텍스트 + 수치)
        fusion_input_dim = text_embed_dim + hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 포지셔널 인코딩
        self.positional_encoding = PositionalEncoding(
            hidden_dim, max_seq_length, dropout
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        멀티모달 임베딩 계산

        Args:
            text_tokens: 텍스트 토큰 (batch_size, seq_len)
            numerical_features: 수치 특성 (batch_size, numerical_dim)

        Returns:
            융합된 임베딩 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = text_tokens.size()

        # 텍스트 임베딩
        text_embed = self.text_embedding(
            text_tokens
        )  # (batch_size, seq_len, text_embed_dim)

        # 수치 데이터 임베딩 및 확장
        numerical_embed = self.numerical_projection(
            numerical_features
        )  # (batch_size, hidden_dim)
        numerical_embed = numerical_embed.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (batch_size, seq_len, hidden_dim)

        # 멀티모달 퓨전
        combined = torch.cat(
            [text_embed, numerical_embed], dim=-1
        )  # (batch_size, seq_len, text_embed_dim + hidden_dim)
        fused = self.fusion_layer(combined)  # (batch_size, seq_len, hidden_dim)

        # 포지셔널 인코딩 추가
        output = self.positional_encoding(fused)

        return output


class PositionalEncoding(nn.Module):
    """포지셔널 인코딩"""

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


class CrossModalAttention(nn.Module):
    """
    크로스 모달 어텐션
    텍스트와 수치 데이터 간의 상호작용 모델링
    """

    def __init__(
        self,
        text_dim: int = 128,
        numerical_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            text_dim: 텍스트 임베딩 차원
            numerical_dim: 수치 임베딩 차원
            hidden_dim: 은닉 차원
            num_heads: 멀티 헤드 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.text_dim = text_dim
        self.numerical_dim = numerical_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Query, Key, Value 프로젝션
        self.W_q_text = nn.Linear(text_dim, hidden_dim)
        self.W_k_num = nn.Linear(numerical_dim, hidden_dim)
        self.W_v_num = nn.Linear(numerical_dim, hidden_dim)

        # Output 프로젝션
        self.W_o = nn.Linear(hidden_dim, text_dim)

        self.dropout = nn.Dropout(dropout)

        assert hidden_dim % num_heads == 0
        self.d_k = hidden_dim // num_heads

    def forward(
        self,
        text_features: torch.Tensor,
        numerical_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        크로스 모달 어텐션 계산

        Args:
            text_features: 텍스트 특성 (batch_size, seq_len, text_dim)
            numerical_features: 수치 특성 (batch_size, seq_len, numerical_dim)
            mask: 어텐션 마스크

        Returns:
            크로스 모달 어텐션 결과 (batch_size, seq_len, text_dim)
        """
        batch_size, seq_len, _ = text_features.size()

        # Query: 텍스트 → Query
        Q = self.W_q_text(text_features)  # (batch_size, seq_len, hidden_dim)

        # Key, Value: 수치 → Key, Value
        K = self.W_k_num(numerical_features)  # (batch_size, seq_len, hidden_dim)
        V = self.W_v_num(numerical_features)  # (batch_size, seq_len, hidden_dim)

        # 멀티 헤드 분할
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 어텐션 스코어
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 어텐션 아웃풋
        output = torch.matmul(attn_weights, V)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        # Output 프로젝션
        output = self.W_o(output)

        return output


class AdaptiveFusionModule(nn.Module):
    """
    적응형 퓨전 모듈
    텍스트와 수치 데이터의 중요도에 따라 동적으로 퓨전
    """

    def __init__(
        self,
        text_dim: int = 128,
        numerical_dim: int = 256,
        hidden_dim: int = 256,
    ):
        """
        Args:
            text_dim: 텍스트 임베딩 차원
            numerical_dim: 수치 임베딩 차원
            hidden_dim: 공유 은닉 차원
        """
        super().__init__()

        self.text_dim = text_dim
        self.numerical_dim = numerical_dim

        # 어텐션 가중치 계산
        self.attention_gate = nn.Sequential(
            nn.Linear(text_dim + numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 출력 퓨전
        self.fusion_layer = nn.Linear(text_dim + numerical_dim, hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        적응형 퓨전

        Args:
            text_features: 텍스트 특성 (batch_size, seq_len, text_dim)
            numerical_features: 수치 특성 (batch_size, seq_len, numerical_dim)

        Returns:
            퓨전된 특성 (batch_size, seq_len, hidden_dim)
        """
        # 어텐션 가중치 계산
        concat = torch.cat([text_features, numerical_features], dim=-1)
        gate = self.attention_gate(concat)  # (batch_size, seq_len, 1)

        # 가중 퓨전
        weighted_text = text_features * gate
        weighted_num = numerical_features * (1 - gate)

        # 최종 퓨전
        fused = torch.cat([weighted_text, weighted_num], dim=-1)
        output = self.fusion_layer(fused)

        return output


if __name__ == "__main__":
    # 테스트
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 모델 생성
    embedding = MultiModalEmbedding(
        vocab_size=5000,
        text_embed_dim=128,
        numerical_dim=5,
        hidden_dim=256,
        max_seq_length=1000,
    ).to(device)

    # 테스트 데이터
    batch_size = 4
    seq_len = 32

    text_tokens = torch.randint(0, 5000, (batch_size, seq_len)).to(device)
    numerical_features = torch.randn(batch_size, 5).to(device)

    # 순전파
    output = embedding(text_tokens, numerical_features)

    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Numerical features shape: {numerical_features.shape}")
    print(f"Output shape: {output.shape}")

    # 크로스 모달 어텐션 테스트
    cross_attn = CrossModalAttention(
        text_dim=128,
        numerical_dim=256,
        hidden_dim=256,
        num_heads=4,
    ).to(device)

    # 텍스트 임베딩 추출
    text_embed = embedding.text_embedding(text_tokens)  # (batch_size, seq_len, 128)
    numerical_embed = (
        embedding.numerical_projection(numerical_features)
        .unsqueeze(1)
        .expand(-1, seq_len, -1)
    )  # (batch_size, seq_len, 256)

    cross_output = cross_attn(text_embed, numerical_embed)
    print(f"Cross-attention output shape: {cross_output.shape}")

    # 적응형 퓨전 테스트
    adaptive_fusion = AdaptiveFusionModule(
        text_dim=128,
        numerical_dim=256,
        hidden_dim=256,
    ).to(device)

    fusion_output = adaptive_fusion(text_embed, numerical_embed)
    print(f"Adaptive fusion output shape: {fusion_output.shape}")
