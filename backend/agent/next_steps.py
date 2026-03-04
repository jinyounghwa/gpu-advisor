"""
Bilingual next-step planner for the 30-day data window workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class NextStep:
    order: int
    id: str
    status: str
    condition: str
    command: Optional[str]
    title_ko: str
    title_en: str
    detail_ko: str
    detail_en: str


def _normalize_readiness(readiness: Dict[str, Any]) -> Dict[str, Any]:
    target_days = int(readiness.get("target_days", 30))
    current_min_days = int(readiness.get("current_min_days", 0))
    remaining_days = int(readiness.get("remaining_days", max(target_days - current_min_days, 0)))
    remaining_days = max(remaining_days, 0)
    ready_for_target = bool(
        readiness.get(
            "ready_for_target",
            readiness.get("ready_for_30d_training", current_min_days >= target_days),
        )
    )
    return {
        "target_days": target_days,
        "current_min_days": current_min_days,
        "remaining_days": remaining_days,
        "ready_for_target": ready_for_target,
    }


def build_post_30d_next_steps(
    readiness: Dict[str, Any],
    release_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = _normalize_readiness(readiness)
    target_days = normalized["target_days"]
    current_days = normalized["current_min_days"]
    remaining_days = normalized["remaining_days"]
    ready = normalized["ready_for_target"]

    release_result = release_result or {}
    release_status = release_result.get("status")
    release_reason = release_result.get("reason")

    if not ready:
        summary_ko = (
            f"아직 {target_days}일 기준에 도달하지 않았습니다. "
            f"남은 {remaining_days}일 동안 일일 수집을 유지하고 릴리즈 드라이런만 수행하세요."
        )
        summary_en = (
            f"The {target_days}-day window is not ready yet. "
            f"Keep daily collection running for the remaining {remaining_days} days and run dry checks only."
        )
        steps = [
            NextStep(
                order=1,
                id="collect_daily_data",
                status="in_progress",
                condition=f"current_min_days={current_days} < target_days={target_days}",
                command="python3 crawlers/run_daily.py",
                title_ko="일일 데이터 수집 유지",
                title_en="Keep Daily Data Collection Running",
                detail_ko="가격/환율/뉴스/피처 데이터가 모두 같은 기간으로 쌓이도록 매일 실행합니다.",
                detail_en="Run the daily crawler so price/exchange/news/feature data keep accumulating on aligned dates.",
            ),
            NextStep(
                order=2,
                id="run_release_dry_check",
                status="ready",
                condition="data window < target",
                command="python3 backend/run_release_daily.py",
                title_ko="릴리즈 드라이 체크 수행",
                title_en="Run Daily Release Dry Check",
                detail_ko="학습 없이 평가/게이트 리포트를 생성해 추세를 미리 점검합니다.",
                detail_en="Generate evaluation/gate reports without training to monitor trend before day-30 readiness.",
            ),
            NextStep(
                order=3,
                id="preflight_release_config",
                status="ready",
                condition="before 30-day unlock",
                command="python3 backend/run_release_ready.py --allow-short-window --no-train --lookback-days 7",
                title_ko="30일 전 사전 점검",
                title_en="Preflight Check Before Day 30",
                detail_ko="릴리즈 실행 경로와 리포트 출력 경로가 정상인지 사전에 검증합니다.",
                detail_en="Validate release runner paths and report generation before the 30-day unlock.",
            ),
            NextStep(
                order=4,
                id="wait_for_30d_unlock",
                status="blocked",
                condition=f"remaining_days={remaining_days}",
                command=None,
                title_ko="30일 게이트 해제 대기",
                title_en="Wait for 30-Day Gate Unlock",
                detail_ko="`current_min_days >= target_days`가 되면 전체 학습+평가 파이프라인을 실행합니다.",
                detail_en="Run the full training+evaluation pipeline once `current_min_days >= target_days`.",
            ),
        ]
    else:
        is_pass = release_status == "pass"
        is_blocked = release_status == "blocked"
        summary_ko = (
            "30일 데이터 윈도우가 확보되었습니다. "
            "이제 학습-평가-릴리즈 게이트를 실행하고 pass/blocked 결과에 따라 후속 단계를 진행하세요."
        )
        summary_en = (
            "The 30-day data window is ready. "
            "Run train-evaluate-release gates and proceed according to pass/blocked outcomes."
        )
        steps = [
            NextStep(
                order=1,
                id="run_release_pipeline",
                status="completed" if release_status in {"pass", "blocked"} else "ready",
                condition=f"current_min_days={current_days} >= target_days={target_days}",
                command="python3 backend/run_release_ready.py",
                title_ko="전체 릴리즈 파이프라인 실행",
                title_en="Run Full Release Pipeline",
                detail_ko="실학습 포함 파이프라인을 실행해 최신 체크포인트와 게이트 판정을 만듭니다.",
                detail_en="Run full pipeline (with training) to produce latest checkpoint and gate decision.",
            ),
            NextStep(
                order=2,
                id="review_release_report",
                status="completed" if release_status in {"pass", "blocked"} else "ready",
                condition="after pipeline run",
                command="cat docs/reports/latest_release_report.md",
                title_ko="최신 릴리즈 리포트 검토",
                title_en="Review Latest Release Report",
                detail_ko="게이트 실패 항목(accuracy/reward/uplift 등)을 확인해 수정 대상을 식별합니다.",
                detail_en="Inspect failed gates (accuracy/reward/uplift, etc.) and identify fix targets.",
            ),
            NextStep(
                order=3,
                id="blocked_retrain_loop",
                status="ready" if is_blocked else "optional",
                condition=f"release_status={release_status or 'not_run'}",
                command="python3 backend/run_release_ready.py --steps 1000 --lookback-days 30",
                title_ko="Blocked 시 재학습 루프",
                title_en="Retraining Loop for Blocked Result",
                detail_ko="blocked이면 스텝/피처/정책 가중치를 조정하고 파이프라인을 재실행합니다.",
                detail_en="If blocked, adjust steps/features/policy weights and re-run the release pipeline.",
            ),
            NextStep(
                order=4,
                id="tag_release_candidate",
                status="ready" if is_pass else "optional",
                condition=f"release_status={release_status or 'not_run'}",
                command="python3 backend/run_release_ready.py --tag --push-tag",
                title_ko="Pass 시 릴리즈 태그 생성",
                title_en="Create Release Tag on Pass",
                detail_ko="게이트가 pass면 태그를 만들고 원격 저장소에 푸시해 배포 후보를 고정합니다.",
                detail_en="If gates pass, create/push a tag to lock a deployable release candidate.",
            ),
        ]

    return {
        "generated_at": datetime.now().isoformat(),
        "target_days": target_days,
        "current_min_days": current_days,
        "remaining_days": remaining_days,
        "ready_for_target": ready,
        "release_status": release_status,
        "release_reason": release_reason,
        "summary_ko": summary_ko,
        "summary_en": summary_en,
        "steps": [asdict(step) for step in steps],
    }

