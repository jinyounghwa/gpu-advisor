"""
GPU Purchase AI Agent
Uses trained world model + MCTS planning to decide buy/wait actions.
"""

from __future__ import annotations

import json
import difflib
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import sys

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.representation_network import RepresentationNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork
from models.mcts_engine import MCTSConfig, MCTSEngine


ACTION_LABELS = {
    0: "BUY_NOW",
    1: "WAIT_SHORT",
    2: "WAIT_LONG",
    3: "HOLD",
    4: "SKIP",
}


@dataclass
class AgentDecision:
    gpu_model: str
    action: str
    raw_action: str
    confidence: float
    entropy: float
    value: float
    action_probs: Dict[str, float]
    expected_rewards: Dict[str, float]
    date: str
    simulations: int
    safe_mode: bool
    safe_reason: str | None


class GPUPurchaseAgent:
    """Planning-based AI agent for GPU purchase decisions."""

    def __init__(
        self,
        project_root: Path | None = None,
        checkpoint_path: Path | None = None,
        num_simulations: int = 50,
        min_confidence: float = 0.25,
        max_entropy: float = 1.58,
    ):
        backend_root = Path(__file__).resolve().parents[1]
        self.project_root = project_root or backend_root.parent
        latest_ckpt = self.project_root / "alphazero_model_agent_latest.pth"
        default_ckpt = latest_ckpt if latest_ckpt.exists() else (self.project_root / "alphazero_model.pth")
        self.checkpoint_path = checkpoint_path or default_ckpt
        self.dataset_dir = self.project_root / "data" / "processed" / "dataset"

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_simulations = num_simulations
        self.min_confidence = min_confidence
        self.max_entropy = max_entropy

        self.representation_network: RepresentationNetwork | None = None
        self.dynamics_network: DynamicsNetwork | None = None
        self.prediction_network: PredictionNetwork | None = None
        self.mcts: MCTSEngine | None = None
        self.model_meta: Dict[str, Any] = {}
        self.action_prior: np.ndarray | None = None

        self._dataset_mtime_ns: int = -1
        self._dataset_rows: List[Dict[str, Any]] = []
        self._models_cache: List[str] = []

        self._load_models()
        self._refresh_dataset_cache(force=True)

    def _load_models(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model_meta = checkpoint.get("meta", {})
        prior = self.model_meta.get("action_prior")
        if isinstance(prior, list) and len(prior) > 0:
            arr = np.asarray(prior, dtype=np.float32)
            if arr.sum() > 0:
                self.action_prior = arr / arr.sum()

        h_state = checkpoint["h_state_dict"]
        g_state = checkpoint["g_state_dict"]
        f_state = checkpoint["f_state_dict"]

        input_dim = h_state["input_embedding.weight"].shape[1]
        latent_dim = h_state["input_embedding.weight"].shape[0]
        action_dim = g_state["input_layer.weight"].shape[1] - latent_dim

        self.representation_network = RepresentationNetwork(
            state_dim=input_dim,
            latent_dim=latent_dim,
        ).to(self.device)
        self.dynamics_network = DynamicsNetwork(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=512,
            num_layers=4,
        ).to(self.device)
        self.prediction_network = PredictionNetwork(
            latent_dim=latent_dim,
            hidden_dim=512,
            num_layers=4,
            action_dim=action_dim,
        ).to(self.device)

        self.representation_network.load_state_dict(h_state)
        self.dynamics_network.load_state_dict(g_state)
        self.prediction_network.load_state_dict(f_state)

        self.representation_network.eval()
        self.dynamics_network.eval()
        self.prediction_network.eval()

        mcts_config = MCTSConfig(num_simulations=self.num_simulations)
        self.mcts = MCTSEngine(mcts_config, latent_dim=latent_dim, action_dim=action_dim)

    def _latest_dataset_path(self) -> Path:
        dated_files = sorted(self.dataset_dir.glob("training_data_*.json"))
        if dated_files:
            return dated_files[-1]
        fallback = self.dataset_dir / "training_data.json"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No processed training dataset file found")

    def _refresh_dataset_cache(self, force: bool = False) -> None:
        dataset_path = self._latest_dataset_path()
        mtime = dataset_path.stat().st_mtime_ns
        if not force and mtime == self._dataset_mtime_ns:
            return
        with open(dataset_path, "r", encoding="utf-8") as f:
            self._dataset_rows = json.load(f)
        self._dataset_mtime_ns = mtime
        self._models_cache = [row["gpu_model"] for row in self._dataset_rows]

    def _resolve_gpu_model(self, query: str) -> str:
        self._refresh_dataset_cache()
        q = query.strip().lower()
        if not q:
            raise ValueError("GPU model name is empty")

        for model in self._models_cache:
            if model.lower() == q:
                return model
        for model in self._models_cache:
            if q in model.lower():
                return model

        matches = difflib.get_close_matches(query, self._models_cache, n=1, cutoff=0.4)
        if matches:
            return matches[0]

        raise ValueError(f"Unknown GPU model: {query}")

    def _get_state_vector(self, gpu_model: str) -> Tuple[np.ndarray, str]:
        self._refresh_dataset_cache()
        for row in self._dataset_rows:
            if row["gpu_model"] == gpu_model:
                return np.asarray(row["state_vector"], dtype=np.float32), row.get("date", "unknown")
        raise ValueError(f"No state vector found for model: {gpu_model}")

    def _one_hot_action(self, action_id: int, action_dim: int) -> torch.Tensor:
        arr = np.zeros(action_dim, dtype=np.float32)
        arr[action_id] = 1.0
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _deterministic_sample(self, probs: np.ndarray, key: str) -> int:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        u = (int(h[:12], 16) % 10_000_000) / 10_000_000.0
        cdf = np.cumsum(probs)
        idx = int(np.searchsorted(cdf, u, side="right"))
        return min(max(idx, 0), len(probs) - 1)

    def decide_from_state(
        self,
        gpu_model: str,
        state_vec: np.ndarray,
        data_date: str,
    ) -> AgentDecision:
        if not all(
            [
                self.representation_network,
                self.dynamics_network,
                self.prediction_network,
                self.mcts,
            ]
        ):
            raise RuntimeError("Agent models not initialized")

        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            latent = self.representation_network(state_tensor).squeeze(0).cpu().numpy()

            mcts_probs_np, root_value, _ = self.mcts.search(
                root_state=latent,
                policy_network=self.prediction_network,
                dynamics_network=self.dynamics_network,
                device=str(self.device),
            )

            expected_rewards: Dict[str, float] = {}
            latent_tensor = torch.tensor(
                latent, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action_dim = len(mcts_probs_np)
            for action_id in range(action_dim):
                action_tensor = self._one_hot_action(action_id, action_dim)
                _, reward_mean, _ = self.dynamics_network(latent_tensor, action_tensor)
                expected_rewards[
                    ACTION_LABELS.get(action_id, f"ACTION_{action_id}")
                ] = float(reward_mean.item())

        # Policy calibration: blend MCTS policy + reward-based policy + learned prior
        reward_arr = np.asarray(
            [
                expected_rewards[ACTION_LABELS.get(i, f"ACTION_{i}")]
                for i in range(len(mcts_probs_np))
            ],
            dtype=np.float32,
        )
        reward_centered = reward_arr - reward_arr.mean()
        reward_scale = max(float(reward_centered.std()), 1e-6)
        reward_scaled = reward_centered / reward_scale
        reward_policy = np.exp(reward_scaled / 1.2)
        reward_policy = reward_policy / reward_policy.sum()

        if self.action_prior is not None and len(self.action_prior) == len(mcts_probs_np):
            prior_policy = self.action_prior.copy()
        else:
            prior_policy = np.ones_like(mcts_probs_np, dtype=np.float32) / len(mcts_probs_np)

        # Sanity layer from observable features (feature_engineer index convention).
        # [0]=price_norm, [1]=ma7, [2]=ma14, [4]=change_1d, [5]=change_7d
        p = float(state_vec[0]) if len(state_vec) > 0 else 0.0
        ma7 = float(state_vec[1]) if len(state_vec) > 1 else p
        ma14 = float(state_vec[2]) if len(state_vec) > 2 else p
        ch1 = float(state_vec[4]) if len(state_vec) > 4 else 0.0
        ch7 = float(state_vec[5]) if len(state_vec) > 5 else 0.0
        over_ma = max((p - 0.5 * (ma7 + ma14)), -0.2)
        trend_up = max(ch1 + 0.5 * ch7, 0.0)
        trend_down = max(-(ch1 + 0.5 * ch7), 0.0)

        utility_bias = np.zeros_like(mcts_probs_np, dtype=np.float32)
        if len(utility_bias) >= 5:
            utility_bias[0] = +1.0 * trend_down - 1.2 * max(over_ma, 0.0) - 0.7 * trend_up  # BUY_NOW
            utility_bias[1] = +0.6 * max(over_ma, 0.0) + 0.5 * trend_up  # WAIT_SHORT
            utility_bias[2] = +0.9 * max(over_ma - 0.02, 0.0) + 0.4 * trend_up  # WAIT_LONG
            utility_bias[3] = +0.2 * (trend_up + trend_down)  # HOLD
            utility_bias[4] = +0.8 * max(over_ma - 0.05, 0.0)  # SKIP
        util_policy = np.exp(utility_bias / 0.8)
        util_policy = util_policy / util_policy.sum()

        calibrated = (
            0.45 * mcts_probs_np
            + 0.25 * reward_policy
            + 0.15 * prior_policy
            + 0.15 * util_policy
        )
        calibrated = np.maximum(calibrated, 0.02)
        calibrated = calibrated / calibrated.sum()

        # Anti-collapse regularizer: enforce minimum action entropy.
        min_entropy_target = 0.65
        ent_now = float(-(calibrated * np.log(calibrated + 1e-10)).sum())
        if ent_now < min_entropy_target:
            alpha = min(0.55, (min_entropy_target - ent_now) / max(min_entropy_target, 1e-6))
            calibrated = (1.0 - alpha) * calibrated + alpha * prior_policy
            calibrated = np.maximum(calibrated, 0.02)
            calibrated = calibrated / calibrated.sum()

        argmax_idx = int(np.argmax(calibrated))
        argmax_conf = float(calibrated[argmax_idx])
        if argmax_conf < 0.75:
            sampled_idx = self._deterministic_sample(calibrated, f"{gpu_model}|{data_date}")
            best_action_idx = sampled_idx
        else:
            best_action_idx = argmax_idx

        raw_action = ACTION_LABELS.get(best_action_idx, f"ACTION_{best_action_idx}")
        confidence = float(calibrated[best_action_idx])
        entropy = float(-(calibrated * np.log(calibrated + 1e-10)).sum())
        safe_mode = False
        safe_reason = None
        action = raw_action

        if confidence < self.min_confidence:
            safe_mode = True
            safe_reason = f"low_confidence<{self.min_confidence:.2f}"
            action = "HOLD"
        elif entropy > self.max_entropy:
            safe_mode = True
            safe_reason = f"high_entropy>{self.max_entropy:.2f}"
            action = "HOLD"

        action_probs = {
            ACTION_LABELS.get(i, f"ACTION_{i}"): float(calibrated[i])
            for i in range(len(calibrated))
        }

        return AgentDecision(
            gpu_model=gpu_model,
            action=action,
            raw_action=raw_action,
            confidence=confidence,
            entropy=entropy,
            value=float(root_value),
            action_probs=action_probs,
            expected_rewards=expected_rewards,
            date=data_date,
            simulations=self.num_simulations,
            safe_mode=safe_mode,
            safe_reason=safe_reason,
        )

    def decide(self, query_model: str) -> AgentDecision:
        resolved_model, state_vec, data_date = self.resolve_state(query_model)
        return self.decide_from_state(resolved_model, state_vec, data_date)

    def resolve_state(self, query_model: str) -> Tuple[str, np.ndarray, str]:
        """Resolve query model and load the latest persisted state vector."""
        resolved_model = self._resolve_gpu_model(query_model)
        state_vec, data_date = self._get_state_vector(resolved_model)
        return resolved_model, state_vec, data_date

    def explain(self, decision: AgentDecision) -> str:
        if decision.action == "BUY_NOW":
            prefix = "지금 매수가 상대적으로 유리하다고 판단했습니다."
        elif decision.action == "WAIT_SHORT":
            prefix = "단기 대기가 더 유리하다고 판단했습니다."
        elif decision.action == "WAIT_LONG":
            prefix = "중기 대기가 더 유리하다고 판단했습니다."
        elif decision.action == "HOLD":
            prefix = "즉시 행동보다 관망이 낫다고 판단했습니다."
        else:
            prefix = "이번 사이클에서는 매수 회피가 유리하다고 판단했습니다."

        return (
            f"{prefix} (신뢰도 {decision.confidence * 100:.1f}%, "
            f"상태가치 {decision.value:.3f}, MCTS {decision.simulations}회)"
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "num_simulations": self.num_simulations,
            "min_confidence": self.min_confidence,
            "max_entropy": self.max_entropy,
            "meta": self.model_meta,
        }
