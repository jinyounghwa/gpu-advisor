"""
Backtest evaluator for GPU purchase agent.
Evaluates day(t) decision against day(t+1) realized price movement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from .gpu_purchase_agent import GPUPurchaseAgent


class AgentEvaluator:
    def __init__(self, project_root: Path, agent: GPUPurchaseAgent):
        self.project_root = project_root
        self.agent = agent
        self.dataset_dir = project_root / "data" / "processed" / "dataset"
        self.raw_danawa_dir = project_root / "data" / "raw" / "danawa"

    def _load_processed(self) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for f in sorted(self.dataset_dir.glob("training_data_*.json")):
            date = f.stem.replace("training_data_", "")
            with open(f, "r", encoding="utf-8") as fp:
                rows = json.load(fp)
            out[date] = {
                row["gpu_model"]: np.asarray(row["state_vector"], dtype=np.float32)
                for row in rows
            }
        return out

    def _load_prices(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for f in sorted(self.raw_danawa_dir.glob("*.json")):
            date = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                out[date] = {
                    p["chipset"]: float(p["lowest_price"])
                    for p in data.get("products", [])
                }
            except Exception:
                continue
        return out

    @staticmethod
    def _reward_for_action(action: str, pct_change: float) -> float:
        if action == "BUY_NOW":
            return pct_change
        if action in {"WAIT_SHORT", "WAIT_LONG"}:
            return -pct_change
        if action == "HOLD":
            return -abs(pct_change) * 0.1
        if action == "SKIP":
            return -abs(pct_change) * 0.15
        return 0.0

    def run(self, lookback_days: int = 30) -> Dict[str, Any]:
        processed = self._load_processed()
        prices = self._load_prices()
        dates = sorted(set(processed.keys()) & set(prices.keys()))
        if len(dates) < 2:
            raise ValueError("Need at least 2 aligned days for evaluation")

        if lookback_days > 0:
            dates = dates[-(lookback_days + 1) :]

        total = 0
        correct_buy_vs_wait = 0
        correct_buy_vs_wait_raw = 0
        cumulative_reward = 0.0
        cumulative_reward_raw = 0.0
        action_counts: Dict[str, int] = {}
        raw_action_counts: Dict[str, int] = {}
        abstain_count = 0
        safe_override_count = 0
        confidence_values: List[float] = []
        all_rewards: List[float] = []
        all_rewards_raw: List[float] = []
        baseline_buy_rewards: List[float] = []
        baseline_wait_rewards: List[float] = []
        baseline_hold_rewards: List[float] = []
        baseline_buy_acc = 0
        baseline_wait_acc = 0

        for i in range(len(dates) - 1):
            d0, d1 = dates[i], dates[i + 1]
            states = processed[d0]
            p0 = prices[d0]
            p1 = prices[d1]
            shared = sorted(set(states.keys()) & set(p0.keys()) & set(p1.keys()))
            for model in shared:
                delta = (p1[model] - p0[model]) / max(p0[model], 1.0)
                decision = self.agent.decide_from_state(model, states[model], d0)
                reward_raw = self._reward_for_action(decision.raw_action, float(delta))
                reward = self._reward_for_action(decision.action, float(delta))
                cumulative_reward_raw += reward_raw
                cumulative_reward += reward
                all_rewards_raw.append(reward_raw)
                all_rewards.append(reward)
                total += 1
                confidence_values.append(decision.confidence)
                raw_action_counts[decision.raw_action] = raw_action_counts.get(decision.raw_action, 0) + 1
                action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
                if decision.action in {"HOLD", "SKIP"}:
                    abstain_count += 1
                if decision.safe_mode and decision.action != decision.raw_action:
                    safe_override_count += 1

                pred_buy = decision.action == "BUY_NOW"
                pred_buy_raw = decision.raw_action == "BUY_NOW"
                true_buy = delta > 0
                if pred_buy == true_buy:
                    correct_buy_vs_wait += 1
                if pred_buy_raw == true_buy:
                    correct_buy_vs_wait_raw += 1

                # Baselines
                r_buy = self._reward_for_action("BUY_NOW", float(delta))
                r_wait = self._reward_for_action("WAIT_SHORT", float(delta))
                r_hold = self._reward_for_action("HOLD", float(delta))
                baseline_buy_rewards.append(r_buy)
                baseline_wait_rewards.append(r_wait)
                baseline_hold_rewards.append(r_hold)
                if true_buy:
                    baseline_buy_acc += 1
                if not true_buy:
                    baseline_wait_acc += 1

        if total == 0:
            raise ValueError("No evaluable samples found")

        probs = np.array(list(action_counts.values()), dtype=np.float32) / total
        raw_probs = np.array(list(raw_action_counts.values()), dtype=np.float32) / total
        action_entropy = float(-(probs * np.log(probs + 1e-10)).sum())
        action_entropy_raw = float(-(raw_probs * np.log(raw_probs + 1e-10)).sum())
        mode_collapse = bool(np.max(probs) >= 0.95)
        mode_collapse_raw = bool(np.max(raw_probs) >= 0.95)

        avg_reward = cumulative_reward / total
        avg_reward_raw = cumulative_reward_raw / total
        base_buy = float(np.mean(baseline_buy_rewards))
        base_wait = float(np.mean(baseline_wait_rewards))
        base_hold = float(np.mean(baseline_hold_rewards))

        return {
            "samples": total,
            "date_from": dates[0],
            "date_to": dates[-1],
            "avg_reward_per_decision": avg_reward,
            "avg_reward_per_decision_raw": avg_reward_raw,
            "directional_accuracy_buy_vs_wait": correct_buy_vs_wait / total,
            "directional_accuracy_buy_vs_wait_raw": correct_buy_vs_wait_raw / total,
            "abstain_ratio": abstain_count / total,
            "safe_override_ratio": safe_override_count / total,
            "avg_confidence": float(np.mean(confidence_values)),
            "action_distribution": action_counts,
            "raw_action_distribution": raw_action_counts,
            "action_entropy": action_entropy,
            "action_entropy_raw": action_entropy_raw,
            "mode_collapse": mode_collapse,
            "mode_collapse_raw": mode_collapse_raw,
            "std_reward": float(np.std(all_rewards)),
            "std_reward_raw": float(np.std(all_rewards_raw)),
            "baseline": {
                "always_buy_reward": base_buy,
                "always_wait_reward": base_wait,
                "always_hold_reward": base_hold,
                "always_buy_accuracy": baseline_buy_acc / total,
                "always_wait_accuracy": baseline_wait_acc / total,
            },
            "uplift_vs_always_buy": avg_reward - base_buy,
            "uplift_raw_vs_always_buy": avg_reward_raw - base_buy,
            "uplift_vs_always_wait": avg_reward - base_wait,
            "uplift_raw_vs_always_wait": avg_reward_raw - base_wait,
            "uplift_vs_always_hold": avg_reward - base_hold,
            "uplift_raw_vs_always_hold": avg_reward_raw - base_hold,
        }
