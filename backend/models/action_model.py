"""
Action Model - 행동 인코딩 및 맥락 인식 행동 사전 분포
GPU Purchase Agent의 액션 표현과 사전 분포를 학습하는 모듈.

MuZero 아키텍처에서 행동은 단순 one-hot 벡터로 인코딩되지만,
이 ActionModel은 두 가지를 추가로 제공한다:
  1. ActionEmbeddingLayer: 5D one-hot → 16D 학습 임베딩 (의미적 관계 포착)
  2. ActionPriorNetwork: Latent State → 행동 사전 확률 (맥락 기반 휴리스틱 대체)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActionEmbeddingLayer(nn.Module):
    """
    이산 행동 인덱스를 연속 임베딩 벡터로 변환한다.

    One-hot 인코딩은 모든 행동을 동등하게 취급하지만,
    학습된 임베딩은 BUY_NOW와 WAIT_SHORT가 개념적으로 가깝고
    SKIP과는 멀다는 사실을 데이터에서 학습할 수 있다.

    입력: action_index (batch,) 또는 one-hot (batch, num_actions)
    출력: 임베딩 벡터 (batch, embed_dim)
    """

    def __init__(self, num_actions: int = 5, embed_dim: int = 16):
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_actions, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, action_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_indices: 정수 행동 인덱스 (batch,)
        Returns:
            embeddings: (batch, embed_dim)
        """
        return self.layer_norm(self.embedding(action_indices))

    def embed_onehot(self, action_onehot: torch.Tensor) -> torch.Tensor:
        """
        One-hot 벡터를 임베딩으로 변환한다. (가중 평균)
        Dynamics Network와의 연동 시 사용.

        Args:
            action_onehot: (batch, num_actions)
        Returns:
            embeddings: (batch, embed_dim)
        """
        # (batch, num_actions) @ (num_actions, embed_dim) → (batch, embed_dim)
        return self.layer_norm(action_onehot @ self.embedding.weight)


class ActionPriorNetwork(nn.Module):
    """
    맥락 인식 행동 사전 분포: Latent State → 행동 확률

    현재 gpu_purchase_agent.py에 하드코딩된 utility_bias (lines 269-276)를
    대체하는 학습 가능한 모듈이다.

    하드코딩된 휴리스틱:
        utility_bias[0] = +1.0 * trend_down - 1.2 * over_ma - 0.7 * trend_up  # BUY_NOW
        utility_bias[1] = +0.6 * over_ma + 0.5 * trend_up                      # WAIT_SHORT
        ...

    이 네트워크는 동일한 직관을 latent 표현에서 end-to-end로 학습한다.
    실제 가격 데이터로부터의 행동 레이블을 지도 학습 신호로 사용한다.

    아키텍처:
        256D latent → 128D → 64D → 5D logits
        파라미터 수: ~43K (경량)
    """

    def __init__(self, latent_dim: int = 256, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: (batch, latent_dim)
        Returns:
            action_logits: (batch, num_actions) — 소프트맥스 전 로짓
        """
        return self.net(latent_state)

    def prior_probs(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: (batch, latent_dim)
        Returns:
            action_probs: (batch, num_actions) — 합이 1인 확률 분포
        """
        return torch.softmax(self.forward(latent_state), dim=-1)


class ActionModel(nn.Module):
    """
    GPU Purchase Agent의 Action Model

    ActionEmbeddingLayer와 ActionPriorNetwork를 결합한 통합 모듈.
    에이전트 체크포인트에 'a_state_dict' 키로 저장되며, 없으면 폴백 동작.

    사용 위치:
        - gpu_purchase_agent.py: get_prior()로 calibrated policy 계산
        - fine_tuner.py: 학습 중 h/g/f와 함께 업데이트
        - (선택적) dynamics_network.py: embed_onehot()으로 행동 인코딩

    파라미터 수:
        - ActionEmbeddingLayer: 5×16 + 16 = ~96
        - ActionPriorNetwork: ~43K
        - 합계: ~43K (전체 모델 18.9M 대비 경량)
    """

    def __init__(
        self,
        num_actions: int = 5,
        embed_dim: int = 16,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embedding = ActionEmbeddingLayer(num_actions, embed_dim)
        self.prior_net = ActionPriorNetwork(latent_dim, num_actions)

    def embed(self, action_indices: torch.Tensor) -> torch.Tensor:
        """정수 인덱스 → 학습 임베딩 (batch,) → (batch, embed_dim)"""
        return self.embedding(action_indices)

    def embed_onehot(self, action_onehot: torch.Tensor) -> torch.Tensor:
        """One-hot 벡터 → 학습 임베딩 (batch, num_actions) → (batch, embed_dim)"""
        return self.embedding.embed_onehot(action_onehot)

    def get_prior(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        현재 시장 상태에서의 맥락 인식 행동 사전 확률.
        gpu_purchase_agent.py의 util_policy 자리를 대체한다.

        Args:
            latent_state: (batch, latent_dim) or (latent_dim,)
        Returns:
            probs: (batch, num_actions) or (num_actions,) — 합이 1
        """
        squeezed = latent_state.ndim == 1
        if squeezed:
            latent_state = latent_state.unsqueeze(0)
        probs = self.prior_net.prior_probs(latent_state)
        return probs.squeeze(0) if squeezed else probs

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: (batch, latent_dim)
        Returns:
            prior_logits: (batch, num_actions)
            prior_probs:  (batch, num_actions)
        """
        logits = self.prior_net(latent_state)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ActionModel(num_actions=5, embed_dim=16, latent_dim=256).to(device)

    batch_size = 4
    latent_dim = 256

    # 정수 인덱스 임베딩
    action_indices = torch.tensor([0, 1, 3, 4], device=device)
    embeddings = model.embed(action_indices)
    print(f"Action embeddings shape: {embeddings.shape}")  # (4, 16)

    # one-hot 임베딩
    action_onehot = F.one_hot(action_indices, num_classes=5).float()
    emb_onehot = model.embed_onehot(action_onehot)
    print(f"One-hot embeddings shape: {emb_onehot.shape}")  # (4, 16)

    # 행동 사전 확률
    latent = torch.randn(batch_size, latent_dim, device=device)
    prior_probs = model.get_prior(latent)
    print(f"Prior probs shape: {prior_probs.shape}")  # (4, 5)
    print(f"Prior probs sum:   {prior_probs.sum(dim=-1)}")  # [1., 1., 1., 1.]

    # 단일 상태 (squeeze 처리 확인)
    single_latent = torch.randn(latent_dim, device=device)
    single_prior = model.get_prior(single_latent)
    print(f"Single prior shape: {single_prior.shape}")  # (5,)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
