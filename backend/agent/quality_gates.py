"""
Centralized quality gates for agent evaluation and release.
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class QualityGateConfig:
    min_accuracy: float = 0.55
    min_avg_reward: float = 0.0
    max_abstain_ratio: float = 0.85
    max_safe_override_ratio: float = 0.90
    min_action_entropy: float = 0.25
    min_uplift_vs_buy: float = 0.0


def check_quality_gates(metrics: Dict[str, Any], cfg: QualityGateConfig = None) -> Dict[str, bool]:
    if cfg is None:
        cfg = QualityGateConfig()
        
    return {
        "accuracy_raw": metrics.get("directional_accuracy_buy_vs_wait_raw", 0) >= cfg.min_accuracy,
        "reward_raw": metrics.get("avg_reward_per_decision_raw", -1) > cfg.min_avg_reward,
        "abstain": metrics.get("abstain_ratio", 1) <= cfg.max_abstain_ratio,
        "safe_override": metrics.get("safe_override_ratio", 1.0) <= cfg.max_safe_override_ratio,
        "action_entropy_raw": metrics.get("action_entropy_raw", 0.0) >= cfg.min_action_entropy,
        "uplift_raw_vs_buy": metrics.get("uplift_raw_vs_always_buy", -1e9) >= cfg.min_uplift_vs_buy,
        "no_mode_collapse_raw": not bool(metrics.get("mode_collapse_raw", True)),
    }
