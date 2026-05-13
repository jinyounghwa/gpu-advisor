"""
레거시 벤치마크 학습 로직.

운영 경로(실데이터 기반):
- backend/agent/*
- backend/simple_server.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

from environment.gym_env import SyntheticMarketEnv
from models.transformer_model import PolicyValueNetwork


@dataclass
class TrainingMetrics:
    """학습 성능 지표"""

    step: int
    episode: int
    tps: float
    vram_mb: float
    ram_mb: float
    loss: float
    reward: float
    elapsed_time: float
    predicted_total_time: float
    done: bool = False


class Trainer:
    """학습 및 벤치마킹 클래스"""

    def __init__(self, model: PolicyValueNetwork, env: SyntheticMarketEnv, config: Dict):
        self.model = model
        self.env = env
        self.config = config

        # 옵티마이저
        self.optimizer = optim.Adam(
            model.parameters(), lr=config.get("learning_rate", 1e-4)
        )

        # 손실 함수
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # 장치
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # 메트릭 저장
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = None

    def _get_memory_usage(self) -> tuple:
        """메모리 사용량 측정"""
        # RAM
        ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        # VRAM (MPS는 torch.mps.current_allocated_memory)
        if torch.backends.mps.is_available():
            vram_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        else:
            vram_mb = 0.0

        return vram_mb, ram_mb

    def _select_action(self, state_vector: torch.Tensor) -> tuple:
        """액션 선택 (Policy-Value Network)"""
        # Convert numpy to torch tensor
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float()

        # State tensor shape: (state_dim,) → (1, 1, state_dim)
        # Model expects: (batch_size, seq_len, state_dim)
        if len(state_vector.shape) == 1:
            state_tensor = state_vector.unsqueeze(0).unsqueeze(0).to(self.device)
        elif len(state_vector.shape) == 2:
            state_tensor = state_vector.unsqueeze(0).to(self.device)
        else:
            state_tensor = state_vector.to(self.device)

        # Forward pass
        policy, value = self.model(state_tensor)

        # Sample action from policy
        action_probs = policy.squeeze(0)
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action, value.item()

    def _compute_loss(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> torch.Tensor:
        """
        손실 계산 (간소화된 PPO-like loss)

        실제로는 더 복잡한 알고리즘을 사용하지만,
        벤치마킹 목적이므로 간단하게 구현
        """
        # Convert numpy to torch tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()

        # Current state forward pass: (state_dim,) → (1, 1, state_dim)
        if len(state.shape) == 1:
            state_tensor = state.unsqueeze(0).unsqueeze(0).to(self.device)
        elif len(state.shape) == 2:
            state_tensor = state.unsqueeze(0).to(self.device)
        else:
            state_tensor = state.to(self.device)

        policy, value = self.model(state_tensor)

        # Policy loss (CrossEntropy with target action)
        # policy shape: (batch_size=1, num_actions=2)
        target_action = torch.tensor([action], dtype=torch.long).to(self.device)
        policy_loss = self.policy_loss_fn(policy, target_action)

        # Value loss (MSE with reward as target)
        target_value = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
        value_loss = self.value_loss_fn(value, target_value)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        return total_loss

    def train_step(self, step: int) -> TrainingMetrics:
        """단일 학습 스텝"""
        step_start = time.time()

        # Environment step
        env_state = self.env._get_state()

        # Convert numpy to torch tensor
        state_vector_np = env_state.state_vector
        if isinstance(state_vector_np, np.ndarray):
            state_vector_tensor = torch.from_numpy(state_vector_np).float()
        else:
            state_vector_tensor = state_vector_np

        # Select action
        action, pred_value = self._select_action(state_vector_tensor)

        # Execute action
        next_state, reward, done, info = self.env.step(action)

        # Compute loss
        state_tensor = env_state.state_vector
        next_state_tensor = next_state.state_vector

        loss = self._compute_loss(state_tensor, action, reward, next_state_tensor, done)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Measure time
        step_time = time.time() - step_start
        tps = 1.0 / step_time

        # Memory usage
        vram_mb, ram_mb = self._get_memory_usage()

        # Elapsed time
        if self.start_time is None:
            self.start_time = time.time()
        elapsed_time = time.time() - self.start_time

        # Predicted total time
        if step > 0:
            predicted_total_time = elapsed_time * (self.config["total_steps"] / step)
        else:
            predicted_total_time = 0.0

        # Create metrics
        metrics = TrainingMetrics(
            step=step,
            episode=step // self.env.steps_per_day,
            tps=tps,
            vram_mb=vram_mb,
            ram_mb=ram_mb,
            loss=loss.item(),
            reward=reward,
            elapsed_time=elapsed_time,
            predicted_total_time=predicted_total_time,
            done=done,
        )

        self.metrics_history.append(metrics)

        return metrics

    def reset(self):
        """학습 상태 초기화"""
        self.env.reset()
        self.start_time = None
        self.metrics_history = []

    def run_benchmark(self, num_steps: int = 1000) -> List[TrainingMetrics]:
        """벤치마크 실행"""
        print(f"\n{'=' * 60}")
        print(f"Running Benchmark: {num_steps} steps")
        print(f"{'=' * 60}\n")

        self.reset()
        results = []

        try:
            for step in range(num_steps):
                metrics = self.train_step(step)
                results.append(metrics)

                # Progress log
                if step % 100 == 0:
                    print(
                        f"Step {step:4d} | TPS: {metrics.tps:6.1f} | "
                        f"VRAM: {metrics.vram_mb:7.1f}MB | "
                        f"RAM: {metrics.ram_mb:7.1f}MB | "
                        f"Loss: {metrics.loss:.4f} | "
                        f"Reward: {metrics.reward:6.2f}"
                    )

                # Reset if episode done
                env_state = self.env._get_state()
                if env_state.step >= self.env.total_steps - 1:
                    self.env.reset()

        except KeyboardInterrupt:
            print("\nBenchmark interrupted")

        # Summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[TrainingMetrics]):
        """벤치마크 결과 요약"""
        if not results:
            return

        print(f"\n{'=' * 60}")
        print("Benchmark Summary")
        print(f"{'=' * 60}\n")

        avg_tps = sum(m.tps for m in results) / len(results)
        max_tps = max(m.tps for m in results)
        min_tps = min(m.tps for m in results)

        avg_vram = sum(m.vram_mb for m in results) / len(results)
        max_vram = max(m.vram_mb for m in results)

        avg_ram = sum(m.ram_mb for m in results) / len(results)

        avg_loss = sum(m.loss for m in results) / len(results)

        print(f"Total Steps: {len(results)}")
        print(f"\nTPS (Transactions Per Second):")
        print(f"  Average: {avg_tps:.2f}")
        print(f"  Max:     {max_tps:.2f}")
        print(f"  Min:     {min_tps:.2f}")
        print(f"\nMemory Usage:")
        print(f"  VRAM (Avg): {avg_vram:.1f} MB")
        print(f"  VRAM (Max):  {max_vram:.1f} MB")
        print(f"  RAM (Avg):   {avg_ram:.1f} MB")
        print(f"\nLoss (Avg): {avg_loss:.4f}")
        print(f"\nPredicted Total Time for {self.config['total_steps']} steps:")
        print(f"  {results[-1].predicted_total_time / 3600:.2f} hours")
        print(f"  {results[-1].predicted_total_time / 86400:.2f} days")

        # Success criteria check
        print(f"\n{'=' * 60}")
        print("Success Criteria Check")
        print(f"{'=' * 60}")
        if avg_tps >= 100:
            print(f"✓ Speed: {avg_tps:.2f} TPS (Target: >= 100 TPS) - PASSED")
        else:
            print(f"✗ Speed: {avg_tps:.2f} TPS (Target: >= 100 TPS) - FAILED")

        if avg_vram < 8000:  # Mac M4 기본형 8GB
            print(f"✓ Memory: {avg_vram:.1f}MB VRAM usage - PASSED")
        else:
            print(f"✗ Memory: {avg_vram:.1f}MB VRAM usage - WARNING")

        print(f"{'=' * 60}\n")
