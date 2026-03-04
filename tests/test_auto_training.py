from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from crawlers.auto_training import AutoTrainingConfig, run_auto_training_cycle


def _write_json(path: Path, body: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _prepare_aligned_data(root: Path, start: date, days: int) -> None:
    for i in range(days):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        _write_json(root / "data" / "raw" / "danawa" / f"{d}.json")
        _write_json(root / "data" / "raw" / "exchange" / f"{d}.json")
        _write_json(root / "data" / "raw" / "news" / f"{d}.json")
        _write_json(root / "data" / "processed" / "dataset" / f"training_data_{d}.json", body="[]")


def test_auto_training_cycle_before_target_runs_release_check(tmp_path: Path):
    _prepare_aligned_data(tmp_path, date(2026, 2, 1), days=10)

    calls = {"daily": 0, "ready": 0}

    def _daily_runner(_project_root: Path, _timeout: int):
        calls["daily"] += 1
        return {"status": "blocked", "reports": {"latest_markdown": "daily.md"}}

    def _ready_runner(_project_root: Path, _config: AutoTrainingConfig):
        calls["ready"] += 1
        return {"status": "blocked", "reports": {"latest_markdown": "ready.md"}}

    cfg = AutoTrainingConfig(
        auto_training_enabled=True,
        target_days=30,
        retrain_every_days=7,
        state_path=str(tmp_path / "data" / "processed" / "auto_state.json"),
    )

    out = run_auto_training_cycle(
        project_root=tmp_path,
        config=cfg,
        release_daily_runner=_daily_runner,
        release_ready_runner=_ready_runner,
    )

    assert out["decision"]["action"] == "release_check"
    assert out["decision"]["reason"] == "insufficient_data_window"
    assert calls["daily"] == 1
    assert calls["ready"] == 0
    assert Path(out["automation_reports"]["json_report"]).exists()


def test_auto_training_cycle_retrain_after_interval(tmp_path: Path):
    _prepare_aligned_data(tmp_path, date(2026, 2, 1), days=35)
    state_path = tmp_path / "data" / "processed" / "auto_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_trained_data_date": "2026-02-28",
                "last_action": "train_release",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = {"daily": 0, "ready": 0}

    def _daily_runner(_project_root: Path, _timeout: int):
        calls["daily"] += 1
        return {"status": "blocked", "reports": {"latest_markdown": "daily.md"}}

    def _ready_runner(_project_root: Path, _config: AutoTrainingConfig):
        calls["ready"] += 1
        return {
            "status": "blocked",
            "reason": "quality_gates_failed",
            "training": {"checkpoint": "alphazero_model_agent_latest.pth"},
            "reports": {"latest_markdown": "ready.md"},
        }

    cfg = AutoTrainingConfig(
        auto_training_enabled=True,
        target_days=30,
        retrain_every_days=7,
        state_path=str(state_path),
    )

    out = run_auto_training_cycle(
        project_root=tmp_path,
        config=cfg,
        release_daily_runner=_daily_runner,
        release_ready_runner=_ready_runner,
    )

    assert out["decision"]["action"] == "train_release"
    assert out["decision"]["reason"] == "retrain_interval_reached"
    assert out["decision"]["newly_accumulated_days"] == 7
    assert out["state_after"]["last_trained_data_date"] == "2026-03-07"
    assert calls["daily"] == 0
    assert calls["ready"] == 1
