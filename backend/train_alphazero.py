"""
MCTS + World Model Training Script
AlphaGo 스타일의 Self-Play를 시장 데이터로 학습
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.representation_network import RepresentationNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork


class AlphaZeroTrainer:
    """AlphaGo 스타일 트레이너"""

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 5,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        device: str = "mps",
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = (
            torch.device(device)
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        # Neural Networks
        self.h = RepresentationNetwork(state_dim=latent_dim, latent_dim=latent_dim).to(
            self.device
        )
        self.g = DynamicsNetwork(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.f = PredictionNetwork(
            latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=action_dim
        ).to(self.device)

        # Optimizers
        self.optimizer_h = optim.Adam(self.h.parameters(), lr=learning_rate)
        self.optimizer_g = optim.Adam(self.g.parameters(), lr=learning_rate)
        self.optimizer_f = optim.Adam(self.f.parameters(), lr=learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def train_step(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        policy_target_batch: torch.Tensor = None,
    ):
        """
        단일 학습 스텝 - MuZero 스타일 손실 함수 적용
        """
        state_batch = state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device).squeeze(-1)
        
        # 1. Action One-hot encoding
        action_onehot = F.one_hot(action_batch, num_classes=self.action_dim).float().to(self.device)

        # 2. Representation: h(o) -> s_0
        # 실제로는 state_batch가 raw observation이어야 함. 
        # 여기서는 state_dim=256인 더미 데이터이므로 h의 state_dim도 256으로 맞춰야 함.
        s_0 = self.h(state_batch)

        # 3. Dynamics: g(s_0, a_0) -> s_1, r_1
        s_1_pred, reward_mean, reward_logvar = self.g(s_0, action_onehot)

        # 4. Prediction: f(s_1) -> p_1, v_1
        policy_logits, value_pred = self.f(s_1_pred)

        # 5. Losses
        # A. Reward Loss (Gaussian NLL)
        reward_loss = F.gaussian_nll_loss(reward_mean, reward_batch, torch.exp(reward_logvar))

        # B. Value Loss
        value_loss = self.mse_loss(value_pred, reward_batch) # 단순화: reward를 타겟으로

        # C. Policy Loss (만약 MCTS 타겟이 있다면)
        if policy_target_batch is not None:
            policy_loss = self.ce_loss(policy_logits, policy_target_batch.to(self.device))
        else:
            # 타겟이 없으면 entropy maximization 등으로 대체 가능하지만 여기선 0
            policy_loss = torch.tensor(0.0, device=self.device)

        # D. Dynamics Loss (Next latent state consistency)
        # s_1_pred 가 실제 s_1 (h(next_state_batch)) 과 같아지도록
        with torch.no_grad():
            s_1_real = self.h(next_state_batch.to(self.device))
        dynamics_loss = self.mse_loss(s_1_pred, s_1_real)

        # Total Loss
        total_loss = reward_loss + value_loss + policy_loss + dynamics_loss

        # 6. Backpropagation
        self.optimizer_h.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        
        total_loss.backward()

        self.optimizer_h.step()
        self.optimizer_g.step()
        self.optimizer_f.step()

        return {
            "total": total_loss.item(),
            "reward": reward_loss.item(),
            "value": value_loss.item(),
            "dynamics": dynamics_loss.item()
        }


def create_synthetic_training_data(num_samples: int = 1000):
    """
    합성 학습 데이터 생성
    """
    latent_dim = 256
    action_dim = 5

    # 랜덤하게 정규화된 State
    states = torch.randn(num_samples, latent_dim)
    actions = torch.randint(0, action_dim, (num_samples,))
    rewards = torch.randn(num_samples, 1) * 0.1  # 작은 변동

    # 다음 State (Identity + 노이즈)
    next_states = states + torch.randn(num_samples, latent_dim) * 0.01

    return states, actions, rewards, next_states


def main():
    """학습 실행"""
    print("=" * 60)
    print("AlphaZero Training - Self-Play on Historical Market Data")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 모델 초기화
    trainer = AlphaZeroTrainer(
        latent_dim=256, action_dim=5, hidden_dim=512, learning_rate=1e-4, device=device
    )

    # 파라미터 수
    total_params = (
        sum(p.numel() for p in trainer.h.parameters())
        + sum(p.numel() for p in trainer.g.parameters())
        + sum(p.numel() for p in trainer.f.parameters())
    )
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 합성 데이터 생성
    print("\nGenerating synthetic training data...")
    states, actions, rewards, next_states = create_synthetic_training_data(1000)

    # 학습 루프
    print("\nStarting training loop...")
    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        for i in range(0, len(states), batch_size):
            batch_end = min(i + batch_size, len(states))
            state_batch = states[i:batch_end]
            action_batch = actions[i:batch_end]
            reward_batch = rewards[i:batch_end]
            next_state_batch = next_states[i:batch_end]

            loss_dict = trainer.train_step(
                state_batch, action_batch, reward_batch, next_state_batch
            )

            if (i + batch_size) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Loss: {loss_dict['total']:.4f} (R: {loss_dict['reward']:.4f}, "
                      f"V: {loss_dict['value']:.4f}, D: {loss_dict['dynamics']:.4f})")

    print("\nTraining completed!")

    # 모델 저장
    print("\nSaving model...")
    torch.save(
        {
            "h_state_dict": trainer.h.state_dict(),
            "g_state_dict": trainer.g.state_dict(),
            "f_state_dict": trainer.f.state_dict(),
        },
        "alphazero_model.pth",
    )
    print("Model saved to alphazero_model.pth")


if __name__ == "__main__":
    main()
