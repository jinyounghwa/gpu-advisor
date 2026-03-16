"""
Automatic post-30-day training/release orchestration.

Policy:
- Before target window: run release dry-check only.
- At first target-day readiness: run full training+release pipeline.
- Afterward: retrain when a configured amount of new data is accumulated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from backend.agent import AgentReleasePipeline


ReleaseDailyRunner = Callable[[Path, int], Dict[str, Any]]
ReleaseReadyRunner = Callable[[Path, "AutoTrainingConfig"], Dict[str, Any]]


@dataclass(frozen=True)
class AutoTrainingConfig:
    auto_training_enabled: bool = True
    target_days: int = 30
    retrain_every_days: int = 7
    timeout_sec: int = 5400
    train_steps: int = 500
    train_batch_size: int = 32
    train_learning_rate: float = 1e-4
    train_seed: int = 42
    lookback_days: int = 30
    state_path: str = "data/processed/auto_training_state.json"

    @staticmethod
    def from_env(project_root: Path) -> "AutoTrainingConfig":
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw.strip())
            except ValueError:
                return default

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw.strip())
            except ValueError:
                return default

        state_path = os.getenv("GPU_ADVISOR_AUTO_TRAIN_STATE_PATH", "data/processed/auto_training_state.json").strip()
        if not state_path:
            state_path = "data/processed/auto_training_state.json"

        cfg = AutoTrainingConfig(
            auto_training_enabled=_env_bool("GPU_ADVISOR_AUTO_TRAIN_ENABLED", True),
            target_days=max(2, _env_int("GPU_ADVISOR_AUTO_TRAIN_TARGET_DAYS", 30)),
            retrain_every_days=max(1, _env_int("GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS", 7)),
            timeout_sec=max(60, _env_int("GPU_ADVISOR_AUTO_TRAIN_TIMEOUT_SEC", 5400)),
            train_steps=max(1, _env_int("GPU_ADVISOR_AUTO_TRAIN_STEPS", 500)),
            train_batch_size=max(1, _env_int("GPU_ADVISOR_AUTO_TRAIN_BATCH_SIZE", 32)),
            train_learning_rate=max(1e-8, _env_float("GPU_ADVISOR_AUTO_TRAIN_LR", 1e-4)),
            train_seed=_env_int("GPU_ADVISOR_AUTO_TRAIN_SEED", 42),
            lookback_days=max(1, _env_int("GPU_ADVISOR_AUTO_TRAIN_LOOKBACK_DAYS", 30)),
            state_path=state_path,
        )
        return cfg.with_project_root(project_root)

    def with_project_root(self, project_root: Path) -> "AutoTrainingConfig":
        state_path = Path(self.state_path)
        if not state_path.is_absolute():
            state_path = project_root / state_path
        return AutoTrainingConfig(
            auto_training_enabled=self.auto_training_enabled,
            target_days=self.target_days,
            retrain_every_days=self.retrain_every_days,
            timeout_sec=self.timeout_sec,
            train_steps=self.train_steps,
            train_batch_size=self.train_batch_size,
            train_learning_rate=self.train_learning_rate,
            train_seed=self.train_seed,
            lookback_days=self.lookback_days,
            state_path=str(state_path),
        )


@dataclass(frozen=True)
class AutoTrainingDecision:
    action: str
    reason: str
    target_days: int
    current_min_days: int
    ready_for_target: bool
    dataset_last_date: Optional[str]
    last_trained_data_date: Optional[str]
    newly_accumulated_days: int
    retrain_every_days: int


def _parse_yyyy_mm_dd(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _days_between(start: Optional[str], end: Optional[str]) -> int:
    d0 = _parse_yyyy_mm_dd(start)
    d1 = _parse_yyyy_mm_dd(end)
    if d0 is None or d1 is None:
        return 0
    return max((d1 - d0).days, 0)


def load_automation_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_automation_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_data_readiness(project_root: Path, target_days: int) -> Dict[str, Any]:
    pipeline = AgentReleasePipeline(project_root=project_root)
    return pipeline.check_readiness(target_days=target_days)


def decide_auto_training_action(
    readiness: Dict[str, Any],
    state: Dict[str, Any],
    config: AutoTrainingConfig,
) -> AutoTrainingDecision:
    details = readiness.get("details", {})
    dataset_last_date = details.get("dataset", {}).get("last_date")
    current_min_days = int(readiness.get("current_min_days", 0))
    target_days = int(readiness.get("target_days", config.target_days))
    ready = bool(readiness.get("ready_for_target", current_min_days >= target_days))
    last_trained_data_date = state.get("last_trained_data_date")
    newly_accumulated_days = _days_between(last_trained_data_date, dataset_last_date)

    if not config.auto_training_enabled:
        return AutoTrainingDecision(
            action="release_check",
            reason="auto_training_disabled",
            target_days=target_days,
            current_min_days=current_min_days,
            ready_for_target=ready,
            dataset_last_date=dataset_last_date,
            last_trained_data_date=last_trained_data_date,
            newly_accumulated_days=newly_accumulated_days,
            retrain_every_days=config.retrain_every_days,
        )

    if not ready:
        return AutoTrainingDecision(
            action="release_check",
            reason="insufficient_data_window",
            target_days=target_days,
            current_min_days=current_min_days,
            ready_for_target=ready,
            dataset_last_date=dataset_last_date,
            last_trained_data_date=last_trained_data_date,
            newly_accumulated_days=newly_accumulated_days,
            retrain_every_days=config.retrain_every_days,
        )

    if not dataset_last_date:
        return AutoTrainingDecision(
            action="release_check",
            reason="dataset_last_date_missing",
            target_days=target_days,
            current_min_days=current_min_days,
            ready_for_target=ready,
            dataset_last_date=dataset_last_date,
            last_trained_data_date=last_trained_data_date,
            newly_accumulated_days=newly_accumulated_days,
            retrain_every_days=config.retrain_every_days,
        )

    if not last_trained_data_date:
        return AutoTrainingDecision(
            action="train_release",
            reason="first_training_after_target",
            target_days=target_days,
            current_min_days=current_min_days,
            ready_for_target=ready,
            dataset_last_date=dataset_last_date,
            last_trained_data_date=last_trained_data_date,
            newly_accumulated_days=newly_accumulated_days,
            retrain_every_days=config.retrain_every_days,
        )

    if newly_accumulated_days >= config.retrain_every_days:
        return AutoTrainingDecision(
            action="train_release",
            reason="retrain_interval_reached",
            target_days=target_days,
            current_min_days=current_min_days,
            ready_for_target=ready,
            dataset_last_date=dataset_last_date,
            last_trained_data_date=last_trained_data_date,
            newly_accumulated_days=newly_accumulated_days,
            retrain_every_days=config.retrain_every_days,
        )

    return AutoTrainingDecision(
        action="release_check",
        reason="waiting_for_more_data_for_retrain",
        target_days=target_days,
        current_min_days=current_min_days,
        ready_for_target=ready,
        dataset_last_date=dataset_last_date,
        last_trained_data_date=last_trained_data_date,
        newly_accumulated_days=newly_accumulated_days,
        retrain_every_days=config.retrain_every_days,
    )


def _run_subprocess(cmd: list[str], cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        fragment = text[idx:]
        try:
            obj, _ = decoder.raw_decode(fragment)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    raise ValueError("No JSON object found in subprocess output")


def run_release_daily_check(project_root: Path, timeout_sec: int) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "backend.run_release_daily"]
    rc, stdout, stderr = _run_subprocess(cmd, cwd=project_root, timeout_sec=timeout_sec)
    if rc != 0:
        detail = (stderr or stdout).strip() or f"exit_code={rc}"
        raise RuntimeError(f"run_release_daily failed: {detail}")

    # run_release_daily.py는 마지막 줄에 JSON을 출력하지만, 중간에 다른 출력이 끼어들
    # 수 있으므로 _extract_first_json_object로 통일하여 파싱 일관성 확보.
    payload = _extract_first_json_object(stdout)
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "release daily returned ok=false"))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("release daily result payload is invalid")
    return result


def run_release_ready_training(project_root: Path, config: AutoTrainingConfig) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "backend.run_release_ready",
        "--steps",
        str(config.train_steps),
        "--batch-size",
        str(config.train_batch_size),
        "--lr",
        str(config.train_learning_rate),
        "--seed",
        str(config.train_seed),
        "--lookback-days",
        str(config.lookback_days),
    ]
    rc, stdout, stderr = _run_subprocess(cmd, cwd=project_root, timeout_sec=config.timeout_sec)
    # run_release_ready.py exits with 1 when status is blocked; this is not a subprocess failure.
    if rc not in (0, 1):
        detail = (stderr or stdout).strip() or f"exit_code={rc}"
        raise RuntimeError(f"run_release_ready failed: {detail}")
    parsed = _extract_first_json_object(stdout)
    if not isinstance(parsed, dict) or "status" not in parsed:
        raise ValueError("run_release_ready output does not include a pipeline result object")
    return parsed


def write_auto_training_report(project_root: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%d_%H%M%S")

    reports_dir = project_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    day_dir = reports_dir / date_dir
    day_dir.mkdir(parents=True, exist_ok=True)

    json_path = day_dir / f"auto_training_status_{ts}.json"
    md_path = day_dir / f"auto_training_status_{ts}.md"
    latest_json = reports_dir / "latest_auto_training_status.json"
    latest_md = reports_dir / "latest_auto_training_status.md"

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    decision = payload.get("decision", {})
    readiness = payload.get("readiness", {})
    execution = payload.get("execution", {})
    pipeline_result = payload.get("pipeline_result", {})
    reports = pipeline_result.get("reports", {}) if isinstance(pipeline_result, dict) else {}

    md_lines = [
        "# GPU Advisor 자동 학습/결과물 생성 상태 보고서",
        "",
        f"- 생성시각: {payload.get('generated_at')}",
        f"- 자동화 실행모드: {execution.get('mode')}",
        f"- 실행 스크립트: {execution.get('script')}",
        f"- 파이프라인 상태: {pipeline_result.get('status')}",
        "",
        "## 1) 자동화 결정 (Decision)",
        f"- action: {decision.get('action')}",
        f"- reason: {decision.get('reason')}",
        f"- current_min_days: {decision.get('current_min_days')}",
        f"- target_days: {decision.get('target_days')}",
        f"- ready_for_target: {decision.get('ready_for_target')}",
        f"- newly_accumulated_days: {decision.get('newly_accumulated_days')}",
        f"- retrain_every_days: {decision.get('retrain_every_days')}",
        "",
        "## 2) 데이터 준비도",
        f"- current_min_days: {readiness.get('current_min_days')}",
        f"- remaining_days: {readiness.get('remaining_days')}",
        f"- ready_for_target: {readiness.get('ready_for_target')}",
        "",
        "## 3) 결과물 (Artifacts)",
        f"- checkpoint: {pipeline_result.get('training', {}).get('checkpoint') if isinstance(pipeline_result.get('training'), dict) else 'N/A'}",
        f"- release report (json): {reports.get('json_report') if isinstance(reports, dict) else 'N/A'}",
        f"- release report (md): {reports.get('markdown_report') if isinstance(reports, dict) else 'N/A'}",
        f"- latest release json: {reports.get('latest_json') if isinstance(reports, dict) else 'N/A'}",
        f"- latest release md: {reports.get('latest_markdown') if isinstance(reports, dict) else 'N/A'}",
        "",
        "## 4) State",
        f"- state_path: {payload.get('state_path')}",
        f"- last_trained_data_date(after): {payload.get('state_after', {}).get('last_trained_data_date')}",
        "",
        "## 5) Bilingual Summary",
        f"- KR: {payload.get('summary_ko')}",
        f"- EN: {payload.get('summary_en')}",
    ]
    md_text = "\n".join(md_lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    return {
        "report_dir": str(day_dir),
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_auto_training_cycle(
    project_root: Path,
    config: AutoTrainingConfig,
    release_daily_runner: Optional[ReleaseDailyRunner] = None,
    release_ready_runner: Optional[ReleaseReadyRunner] = None,
) -> Dict[str, Any]:
    release_daily_runner = release_daily_runner or run_release_daily_check
    release_ready_runner = release_ready_runner or run_release_ready_training

    cfg = config.with_project_root(project_root)
    state_path = Path(cfg.state_path)
    state_before = load_automation_state(state_path)
    readiness = get_data_readiness(project_root=project_root, target_days=cfg.target_days)
    decision = decide_auto_training_action(readiness=readiness, state=state_before, config=cfg)

    if decision.action == "train_release":
        pipeline_result = release_ready_runner(project_root, cfg)
        mode = "training_release"
        script = "backend/run_release_ready.py"
    else:
        pipeline_result = release_daily_runner(project_root, cfg.timeout_sec)
        mode = "release_check"
        script = "backend/run_release_daily.py"

    now_iso = datetime.now().isoformat()
    state_after = dict(state_before)
    state_after["updated_at"] = now_iso
    state_after["last_action"] = decision.action
    state_after["last_reason"] = decision.reason
    state_after["last_pipeline_status"] = pipeline_result.get("status")
    state_after["last_decision"] = asdict(decision)
    state_after["last_pipeline_reports"] = pipeline_result.get("reports")
    if decision.action == "train_release" and decision.dataset_last_date:
        state_after["last_trained_data_date"] = decision.dataset_last_date
        state_after["last_training_run_at"] = now_iso

    save_automation_state(state_path, state_after)

    summary_ko = (
        "30일 윈도우 이후 자동 학습/평가 루틴이 실행되었습니다."
        if decision.action == "train_release"
        else "자동 학습 조건 미충족이므로 평가/리포트 루틴만 실행되었습니다."
    )
    summary_en = (
        "Automatic post-30-day training/evaluation routine was executed."
        if decision.action == "train_release"
        else "Training condition was not met, so evaluation/report routine ran only."
    )

    payload: Dict[str, Any] = {
        "generated_at": now_iso,
        "summary_ko": summary_ko,
        "summary_en": summary_en,
        "config": asdict(cfg),
        "readiness": readiness,
        "decision": asdict(decision),
        "execution": {
            "mode": mode,
            "script": script,
        },
        "pipeline_result": pipeline_result,
        "state_path": str(state_path),
        "state_before": state_before,
        "state_after": state_after,
    }
    payload["automation_reports"] = write_auto_training_report(project_root, payload)
    return payload
