"""
Fallback release report writer used when release pipeline execution fails.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def write_fallback_release_report(
    project_root: Path,
    reason: str,
    detail: str,
    run_started_at: str,
) -> Dict[str, str]:
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%d_%H%M%S")

    reports_dir = project_root / "docs" / "reports"
    day_dir = reports_dir / date_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    day_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "generated_at": now.isoformat(),
        "run_started_at": run_started_at,
        "status": "blocked",
        "reason": reason,
        "readiness": None,
        "evaluation": None,
        "gates": None,
        "error": detail,
        "fallback_report": True,
    }

    json_path = day_dir / f"release_report_{ts}.json"
    md_path = day_dir / f"release_report_{ts}.md"
    latest_json = reports_dir / "latest_release_report.json"
    latest_md = reports_dir / "latest_release_report.md"

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    md_text = "\n".join(
        [
            "# GPU Advisor 릴리즈 판정 보고서",
            "",
            f"- 생성시각: {payload['generated_at']}",
            "- 최종판정: **blocked**",
            "- 폴백 리포트: true",
            "",
            "## 1) 사유",
            f"- reason: {reason}",
            "",
            "## 2) 상세",
            f"- detail: {detail}",
        ]
    )
    md_path.write_text(md_text + "\n", encoding="utf-8")
    latest_md.write_text(md_text + "\n", encoding="utf-8")

    return {
        "report_dir": str(day_dir),
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
