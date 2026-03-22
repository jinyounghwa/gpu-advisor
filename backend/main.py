"""
레거시 벤치마크 진입점.

운영 경로(실데이터 기반):
- 수집: crawlers/run_daily.py
- 서버/추론: backend/simple_server.py
- 릴리즈 파이프라인: backend/run_release_ready.py
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.gym_env import SyntheticMarketEnv
from models.transformer_model import create_model
from api.training import Trainer


def main():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    data_path = os.path.join(
        project_root, "data", "processed", "dataset", "training_data.json"
    )

    print(f"Data path: {data_path}")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    env_config = {"data_path": data_path, "total_days": 200}

    env = SyntheticMarketEnv(**env_config)

    print(f"\nEnvironment created:")
    print(f"  Total days: {env.total_days}")
    print(f"  Steps per day: {env.steps_per_day}")
    print(f"  Total steps: {env.total_steps}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")

    model_config = {
        "input_dim": 11,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 1024,
        "num_actions": 2,
    }

    model = create_model(model_config)

    trainer_config = {"learning_rate": 1e-4, "total_steps": env.total_steps}

    trainer = Trainer(model, env, trainer_config)

    trainer.run_benchmark(num_steps=1000)


if __name__ == "__main__":
    main()
