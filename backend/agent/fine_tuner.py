"""
Fine-tuning loop for GPU purchase world model agent.
Uses crawled datasets (processed features + raw prices) to train h/g/f networks.
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .gpu_purchase_agent import ACTION_LABELS
from models.representation_network import RepresentationNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork


@dataclass
class TransitionSample:
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: float
    value_target: float


class AgentFineTuner:
    def __init__(
        self,
        project_root: Path,
        base_checkpoint: Optional[Path] = None,
        output_checkpoint: Optional[Path] = None,
    ):
        self.project_root = project_root
        self.base_checkpoint = base_checkpoint or (project_root / "alphazero_model.pth")
        self.output_checkpoint = output_checkpoint or (project_root / "alphazero_model_agent_latest.pth")
        self.dataset_dir = project_root / "data" / "processed" / "dataset"
        self.raw_danawa_dir = project_root / "data" / "raw" / "danawa"

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.samples: List[TransitionSample] = []
        self.recent_losses: List[float] = []

        self.h: RepresentationNetwork
        self.g: DynamicsNetwork
        self.f: PredictionNetwork
        self.optimizer: torch.optim.Optimizer

        self._load_models()
        self._build_transition_dataset()
        self.dataset_summary = self._summarize_dataset()
        self.class_weights, self.action_prior = self._build_action_stats()

    def _load_models(self) -> None:
        if not self.base_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.base_checkpoint}")
        ckpt = torch.load(self.base_checkpoint, map_location=self.device, weights_only=False)

        h_state = ckpt["h_state_dict"]
        g_state = ckpt["g_state_dict"]
        f_state = ckpt["f_state_dict"]

        input_dim = h_state["input_embedding.weight"].shape[1]
        latent_dim = h_state["input_embedding.weight"].shape[0]
        action_dim = g_state["input_layer.weight"].shape[1] - latent_dim

        self.h = RepresentationNetwork(state_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.g = DynamicsNetwork(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=512, num_layers=4).to(
            self.device
        )
        self.f = PredictionNetwork(latent_dim=latent_dim, hidden_dim=512, num_layers=4, action_dim=action_dim).to(
            self.device
        )

        self.h.load_state_dict(h_state)
        self.g.load_state_dict(g_state)
        self.f.load_state_dict(f_state)

    def _load_processed_by_date(self) -> Dict[str, Dict[str, np.ndarray]]:
        files = sorted(self.dataset_dir.glob("training_data_*.json"))
        if len(files) < 2:
            raise ValueError("Need at least 2 processed daily datasets for transition training")
        by_date: Dict[str, Dict[str, np.ndarray]] = {}
        for f in files:
            date = f.stem.replace("training_data_", "")
            with open(f, "r", encoding="utf-8") as fp:
                rows = json.load(fp)
            model_map: Dict[str, np.ndarray] = {}
            for row in rows:
                model_map[row["gpu_model"]] = np.asarray(row["state_vector"], dtype=np.float32)
            by_date[date] = model_map
        return by_date

    def _load_prices_by_date(self) -> Dict[str, Dict[str, float]]:
        prices: Dict[str, Dict[str, float]] = {}
        for f in sorted(self.raw_danawa_dir.glob("*.json")):
            date = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                products = data.get("products", [])
                prices[date] = {p["chipset"]: float(p["lowest_price"]) for p in products}
            except Exception:
                continue
        return prices

    def _action_from_delta(self, pct_change: float) -> int:
        # Positive future price change means buying now was better than waiting.
        if pct_change >= 0.02:
            return 0  # BUY_NOW
        if pct_change <= -0.05:
            return 4  # SKIP
        if pct_change <= -0.02:
            return 2  # WAIT_LONG
        if pct_change <= -0.005:
            return 1  # WAIT_SHORT
        return 3  # HOLD

    def _build_transition_dataset(self) -> None:
        processed = self._load_processed_by_date()
        prices = self._load_prices_by_date()
        dates = sorted(processed.keys())

        samples: List[TransitionSample] = []
        for i in range(len(dates) - 1):
            d0 = dates[i]
            d1 = dates[i + 1]
            map0 = processed[d0]
            map1 = processed[d1]
            p0 = prices.get(d0, {})
            p1 = prices.get(d1, {})
            shared = sorted(set(map0.keys()) & set(map1.keys()) & set(p0.keys()) & set(p1.keys()))

            for model in shared:
                price0 = max(p0[model], 1.0)
                price1 = p1[model]
                pct_change = (price1 - price0) / price0
                action = self._action_from_delta(pct_change)
                reward = float(np.clip(pct_change, -1.0, 1.0))
                value_target = float(np.tanh(pct_change * 8.0))
                samples.append(
                    TransitionSample(
                        state=map0[model],
                        next_state=map1[model],
                        action=action,
                        reward=reward,
                        value_target=value_target,
                    )
                )

        if not samples:
            raise ValueError("No transition samples built from current datasets")
        self.samples = samples

    def _summarize_dataset(self) -> Dict[str, float]:
        rewards = [s.reward for s in self.samples]
        actions = [s.action for s in self.samples]
        return {
            "num_samples": len(self.samples),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "buy_ratio": float(np.mean(np.asarray(actions) == 0)),
        }

    def _build_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        actions = np.asarray([s.action for s in self.samples], dtype=np.int64)
        counts = np.bincount(actions, minlength=5).astype(np.float32)
        counts = np.clip(counts, 1.0, None)
        inv = 1.0 / counts
        class_weights = inv / inv.sum() * len(inv)
        prior = counts / counts.sum()
        return (
            torch.tensor(class_weights, dtype=torch.float32, device=self.device),
            torch.tensor(prior, dtype=torch.float32, device=self.device),
        )

    def _memory_usage(self) -> tuple[float, float]:
        ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.backends.mps.is_available():
            vram_mb = torch.mps.current_allocated_memory() / 1024 / 1024
        else:
            vram_mb = 0.0
        return vram_mb, ram_mb

    def _sample_batch(self, batch_size: int) -> List[TransitionSample]:
        idx = np.random.choice(len(self.samples), size=min(batch_size, len(self.samples)), replace=True)
        return [self.samples[i] for i in idx]

    def _train_step(self, batch: List[TransitionSample]) -> Dict[str, float]:
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        value_targets = torch.tensor([b.value_target for b in batch], dtype=torch.float32, device=self.device)

        latent = self.h(states)
        with torch.no_grad():
            latent_next_target = self.h(next_states)

        policy_logits, value_pred = self.f(latent)
        action_onehot = F.one_hot(actions, num_classes=5).float()
        latent_next_pred, reward_pred, _ = self.g(latent, action_onehot)

        policy_loss = F.cross_entropy(policy_logits, actions, weight=self.class_weights)
        value_loss = F.mse_loss(value_pred, value_targets)
        dynamics_loss = F.mse_loss(latent_next_pred, latent_next_target)
        reward_loss = F.mse_loss(reward_pred, rewards)

        probs = torch.softmax(policy_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        # Encourage policy distribution to stay close to empirical action prior.
        mean_probs = probs.mean(dim=0)
        prior_reg = F.kl_div(
            torch.log(mean_probs + 1e-10),
            self.action_prior,
            reduction="batchmean",
        )

        total_loss = (
            policy_loss
            + value_loss
            + dynamics_loss
            + reward_loss
            - 0.001 * entropy
            + 0.02 * prior_reg
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.h.parameters()) + list(self.g.parameters()) + list(self.f.parameters()), 1.0)
        self.optimizer.step()

        pred_actions = probs.argmax(dim=-1)
        acc = (pred_actions == actions).float().mean().item()
        avg_probs = probs.mean(dim=0).detach().cpu().numpy().tolist()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "dynamics_loss": float(dynamics_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "entropy": float(entropy.item()),
            "accuracy": float(acc),
            "action_probs": avg_probs,
            "reward": float(rewards.mean().item()),
            "prior_reg": float(prior_reg.item()),
        }

    def run(
        self,
        num_steps: int,
        batch_size: int,
        learning_rate: float,
        on_step: Callable[[Dict], None],
        should_stop: Callable[[], bool],
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        self.h.train()
        self.g.train()
        self.f.train()

        params = list(self.h.parameters()) + list(self.g.parameters()) + list(self.f.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-5)
        start_time = time.time()

        for step in range(1, num_steps + 1):
            if should_stop():
                break
            t0 = time.time()
            batch = self._sample_batch(batch_size)
            out = self._train_step(batch)
            dt = max(time.time() - t0, 1e-6)
            tps = 1.0 / dt
            vram_mb, ram_mb = self._memory_usage()

            metric = {
                "step": step,
                "episode": step // 50,
                "timestamp": time.time(),
                "elapsed_time": time.time() - start_time,
                "loss": round(out["loss"], 4),
                "policy_loss": round(out["policy_loss"], 4),
                "value_loss": round(out["value_loss"], 4),
                "reward": round(out["reward"], 4),
                "entropy": round(out["entropy"], 4),
                "prior_reg": round(out["prior_reg"], 4),
                "tps": round(tps, 1),
                "vram_mb": round(vram_mb, 1),
                "ram_mb": round(ram_mb, 1),
                "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
                "learning_rate": learning_rate,
                "grad_norm": 0.0,
                "win_rate": round(out["accuracy"], 4),
                "episode_length": len(batch),
                "action_probs": [round(x, 4) for x in out["action_probs"]],
                "progress": round(step / max(num_steps, 1) * 100, 1),
            }
            on_step(metric)

        self.save_checkpoint(
            {
                "seed": seed,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )

    def save_checkpoint(self, train_config: Optional[Dict[str, float]] = None) -> None:
        payload = {
            "h_state_dict": self.h.state_dict(),
            "g_state_dict": self.g.state_dict(),
            "f_state_dict": self.f.state_dict(),
            "meta": {
                "source": "agent_finetuner",
                "action_labels": ACTION_LABELS,
                "num_samples": len(self.samples),
                "saved_at": time.time(),
                "train_config": train_config or {},
                "dataset_summary": self.dataset_summary,
                "action_prior": [float(x) for x in self.action_prior.detach().cpu().numpy().tolist()],
                "schema_version": "agent-v1",
            },
        }
        torch.save(payload, self.output_checkpoint)
