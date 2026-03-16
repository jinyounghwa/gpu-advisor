#!/usr/bin/env python3
"""
Run daily release check and print machine-readable JSON result.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from backend.agent import AgentReleasePipeline, PipelineConfig

# MCTS 실행 예산: pairs × GPUs_est × sims ≤ TARGET_OPS
_TARGET_OPS = 5000
_MAX_LOOKBACK = 14       # 최대 2주 롤링 윈도우
_GPU_COUNT_EST = 22      # 평균 GPU 모델 수 추정치
_MIN_SIMS = 10           # 최소 시뮬레이션 수


def _adaptive_config(dataset_dir: Path) -> tuple[int, int]:
    """사용 가능한 데이터 일수에 따라 lookback_days, num_simulations 자동 결정."""
    available_days = len([
        f for f in dataset_dir.glob("training_data_*.json")
        if f.stem.replace("training_data_", "").count("-") == 2
    ])
    lookback = min(available_days, _MAX_LOOKBACK)
    pairs = max(1, min(available_days - 1, _MAX_LOOKBACK - 1))
    num_sims = max(_MIN_SIMS, _TARGET_OPS // (pairs * _GPU_COUNT_EST))
    return lookback, num_sims


def main() -> int:
    try:
        dataset_dir = ROOT / "data" / "processed" / "dataset"
        lookback_days, num_simulations = _adaptive_config(dataset_dir)
        pipeline = AgentReleasePipeline(project_root=ROOT)
        cfg = PipelineConfig(
            run_training=False,
            require_30d=False,
            lookback_days=lookback_days,
            num_simulations=num_simulations,
        )
        result = pipeline.run(cfg)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
