"""
합성 시뮬레이션 환경 (Synthetic Environment)

주의:
- 본 모듈은 레거시 벤치마크용입니다.
- 운영 경로(실데이터 기반 에이전트)는 backend/simple_server.py + backend/agent/* 를 사용합니다.
"""

import numpy as np
import json
from typing import Tuple
from dataclasses import dataclass


@dataclass
class EnvironmentState:
    state_vector: np.ndarray
    step: int
    total_steps: int


class SyntheticMarketEnv:
    """합성 시장 환경 - Gym 인터페이스 (레거시 벤치마크 전용)"""

    def __init__(self, data_path: str, total_days: int = 200):
        """
        Args:
            data_path: 3일치 데이터 경로
            total_days: 시뮬레이션할 총 일수
        """
        self.total_days = total_days
        self.steps_per_day = 100  # 하루당 스텝 수
        self.total_steps = total_days * self.steps_per_day
        self.current_step = 0

        # 3일치 데이터 로드
        with open(data_path, "r") as f:
            self.sample_data = json.load(f)

        # state_vector dimension 추출
        self.state_dim = len(self.sample_data[0]["state_vector"])
        self.action_dim = 2  # 0: 관망, 1: 구매

        # 데이터 확장 (3일 → 200일)
        self.simulated_data = self._expand_data()

    def _expand_data(self) -> list:
        """3일치 데이터를 200일로 확장 (Augmentation + Loop)"""
        expanded = []
        num_samples = len(self.sample_data)

        for day in range(self.total_days):
            for step in range(self.steps_per_day):
                # 원본 데이터 루핑 + 약간의 노이즈 추가
                original_idx = step % num_samples
                sample = self.sample_data[original_idx].copy()

                # 약간의 랜덤 노이즈 추가 (데이터 다양성)
                noise = np.random.normal(0, 0.01, self.state_dim)
                noisy_state = np.array(sample["state_vector"]) + noise
                noisy_state = np.clip(noisy_state, 0, 1)  # [0, 1] 범위 유지

                expanded.append(
                    {
                        "date": sample["date"],
                        "day": day,
                        "step": step,
                        "gpu_model": sample["gpu_model"],
                        "state_vector": noisy_state.tolist(),
                    }
                )

        return expanded

    def reset(self) -> EnvironmentState:
        """환경 초기화"""
        self.current_step = 0
        return self._get_state()

    def _get_state(self) -> EnvironmentState:
        """현재 상태 반환"""
        data = self.simulated_data[self.current_step]
        state_vector = np.array(data["state_vector"], dtype=np.float32)

        return EnvironmentState(
            state_vector=state_vector,
            step=self.current_step,
            total_steps=self.total_steps,
        )

    def step(self, action: int) -> Tuple[EnvironmentState, float, bool, dict]:
        """
        Action 수행 및 다음 상태, 리워드 반환

        Args:
            action: 0 (관망), 1 (구매)

        Returns:
            state: 다음 상태
            reward: 리워드 (수익률)
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        # 현재 상태
        current_data = self.simulated_data[self.current_step]

        # 리워드 계산 (단순화: 구매 시 현재 가격 변동 반영)
        if action == 1:  # 구매
            # state_vector[0]이 가격 변동률이라 가정
            price_change = current_data["state_vector"][0]
            reward = price_change * 100  # 수익률 스케일링
        else:  # 관망
            reward = 0.0

        # 다음 스텝으로 이동
        self.current_step += 1

        # 종료 조건
        done = self.current_step >= self.total_steps - 1

        # 다음 상태
        next_state = self._get_state()

        info = {
            "day": current_data["day"],
            "step": current_data["step"],
            "gpu_model": current_data["gpu_model"],
            "price_change": current_data["state_vector"][0],
        }

        return next_state, reward, done, info

    @property
    def observation_space(self):
        return self.state_dim

    @property
    def action_space(self):
        return self.action_dim


# Backward compatibility for legacy imports.
DummyMarketEnv = SyntheticMarketEnv
