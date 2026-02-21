"""
Release pipeline for production-candidate GPU purchase agent.
Stages: readiness -> training -> evaluation -> quality gates -> report artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .fine_tuner import AgentFineTuner
from .gpu_purchase_agent import GPUPurchaseAgent
from .evaluator import AgentEvaluator


@dataclass
class PipelineConfig:
    target_days: int = 30
    lookback_days: int = 30
    num_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    min_accuracy: float = 0.55
    min_avg_reward: float = 0.0
    max_abstain_ratio: float = 0.85
    max_safe_override_ratio: float = 0.90
    min_action_entropy: float = 0.25
    min_uplift_vs_buy: float = 0.0
    require_30d: bool = True
    run_training: bool = True


class AgentReleasePipeline:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "docs" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_readiness(self, target_days: int = 30) -> Dict[str, Any]:
        from datetime import datetime as _dt

        targets = {
            "danawa": self.project_root / "data" / "raw" / "danawa",
            "exchange": self.project_root / "data" / "raw" / "exchange",
            "news": self.project_root / "data" / "raw" / "news",
            "dataset": self.project_root / "data" / "processed" / "dataset",
        }
        details: Dict[str, Any] = {}
        min_days = None
        for name, root in targets.items():
            dates = []
            for f in sorted(root.glob("*.json")):
                stem = f.stem.replace("training_data_", "")
                try:
                    dates.append(_dt.strptime(stem, "%Y-%m-%d").date())
                except ValueError:
                    continue
            dates = sorted(set(dates))
            range_days = (dates[-1] - dates[0]).days + 1 if dates else 0
            details[name] = {
                "dated_files": len(dates),
                "range_days": range_days,
                "first_date": str(dates[0]) if dates else None,
                "last_date": str(dates[-1]) if dates else None,
            }
            min_days = range_days if min_days is None else min(min_days, range_days)
        min_days = min_days or 0
        return {
            "target_days": target_days,
            "current_min_days": min_days,
            "remaining_days": max(target_days - min_days, 0),
            "ready_for_target": min_days >= target_days,
            "details": details,
        }

    def train(self, cfg: PipelineConfig) -> Dict[str, Any]:
        trainer = AgentFineTuner(project_root=self.project_root)
        latest_metric: Dict[str, Any] = {}

        def on_step(metric: Dict[str, Any]) -> None:
            nonlocal latest_metric
            latest_metric = metric

        def should_stop() -> bool:
            return False

        trainer.run(
            num_steps=cfg.num_steps,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            on_step=on_step,
            should_stop=should_stop,
            seed=cfg.seed,
        )
        return {
            "status": "completed",
            "latest_metric": latest_metric,
            "checkpoint": str(self.project_root / "alphazero_model_agent_latest.pth"),
        }

    def evaluate(self, cfg: PipelineConfig) -> Dict[str, Any]:
        agent = GPUPurchaseAgent(project_root=self.project_root)
        evaluator = AgentEvaluator(project_root=self.project_root, agent=agent)
        metrics = evaluator.run(lookback_days=cfg.lookback_days)
        return metrics

    def quality_gates(self, metrics: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, bool]:
        return {
            "accuracy_raw": metrics["directional_accuracy_buy_vs_wait_raw"] >= cfg.min_accuracy,
            "reward_raw": metrics["avg_reward_per_decision_raw"] > cfg.min_avg_reward,
            "abstain": metrics["abstain_ratio"] <= cfg.max_abstain_ratio,
            "safe_override": metrics.get("safe_override_ratio", 1.0) <= cfg.max_safe_override_ratio,
            "action_entropy_raw": metrics.get("action_entropy_raw", 0.0) >= cfg.min_action_entropy,
            "uplift_raw_vs_buy": metrics.get("uplift_raw_vs_always_buy", -1e9) >= cfg.min_uplift_vs_buy,
            "no_mode_collapse_raw": not bool(metrics.get("mode_collapse_raw", True)),
        }

    def write_report(self, payload: Dict[str, Any]) -> Dict[str, str]:
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        ts = now.strftime("%Y%m%d_%H%M%S")
        day_dir = self.reports_dir / date_dir
        day_dir.mkdir(parents=True, exist_ok=True)

        json_path = day_dir / f"release_report_{ts}.json"
        md_path = day_dir / f"release_report_{ts}.md"
        latest_json = self.reports_dir / "latest_release_report.json"
        latest_md = self.reports_dir / "latest_release_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        readiness = payload.get("readiness", {})
        status = payload.get("status")
        evalm = payload.get("evaluation", {})
        gates = payload.get("gates", {})
        def _v(key: str) -> Any:
            val = evalm.get(key)
            return "N/A" if val is None else val
        def _g(key: str) -> Any:
            val = gates.get(key)
            return "N/A" if val is None else val

        gate_lines = []
        if gates:
            for k, v in gates.items():
                gate_lines.append(f"- {k}: {_g(k)}")
        else:
            gate_lines.append("- N/A")

        md = [
            "# GPU Advisor 릴리즈 판정 보고서",
            "",
            f"- 생성시각: {datetime.now().isoformat()}",
            f"- 최종판정: **{status}**",
            "",
            "## 1) 데이터 준비도",
            f"- 목표 일수: {readiness.get('target_days')}",
            f"- 최소 확보 일수: {readiness.get('current_min_days')}",
            f"- 잔여 일수: {readiness.get('remaining_days')}",
            "",
            "## 2) 평가 지표",
            f"- 표본수: {_v('samples')}",
            f"- 방향정확도(BUY vs WAIT): {_v('directional_accuracy_buy_vs_wait')}",
            f"- 의사결정당 평균 보상: {_v('avg_reward_per_decision')}",
            f"- 관망비율: {_v('abstain_ratio')}",
            f"- 평균 신뢰도: {_v('avg_confidence')}",
            "",
            "## 3) 게이트 통과 여부",
            *gate_lines,
            "",
            "## 4) 결론",
            "- `pass`면 공개 후보, `blocked`면 데이터/학습/정책 재조정 필요.",
        ]
        md_text = "\n".join(md)
        md_path.write_text(md_text, encoding="utf-8")
        latest_md.write_text(md_text, encoding="utf-8")
        return {
            "report_dir": str(day_dir),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }

    def run(self, cfg: Optional[PipelineConfig] = None) -> Dict[str, Any]:
        cfg = cfg or PipelineConfig()
        readiness = self.check_readiness(target_days=cfg.target_days)
        result: Dict[str, Any] = {
            "config": asdict(cfg),
            "readiness": readiness,
            "status": "blocked",
        }

        # 최소 2일 전이는 있어야 평가 자체가 의미 있음(상태 전이/보상 계산 불가 방지)
        if readiness["current_min_days"] < 2:
            result["reason"] = "insufficient_data_window"
            result["reports"] = self.write_report(result)
            return result

        if cfg.require_30d and not readiness["ready_for_target"]:
            result["reason"] = "insufficient_data_window"
            result["reports"] = self.write_report(result)
            return result

        if cfg.run_training:
            result["training"] = self.train(cfg)

        evaluation = self.evaluate(cfg)
        gates = self.quality_gates(evaluation, cfg)
        result["evaluation"] = evaluation
        result["gates"] = gates
        result["status"] = "pass" if all(gates.values()) else "blocked"
        if result["status"] != "pass":
            result["reason"] = "quality_gates_failed"
        result["reports"] = self.write_report(result)
        return result
