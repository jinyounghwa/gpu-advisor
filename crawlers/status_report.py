"""
Daily data status report generator.
Creates machine-readable and markdown summaries under docs/reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _collect_date_summary(root: Path, prefix_to_strip: str = "") -> Dict[str, Any]:
    dates = []
    if root.exists():
        for file in sorted(root.glob("*.json")):
            stem = file.stem
            if prefix_to_strip and stem.startswith(prefix_to_strip):
                stem = stem.replace(prefix_to_strip, "", 1)
            try:
                dt = datetime.strptime(stem, "%Y-%m-%d").date()
                dates.append(dt)
            except ValueError:
                continue

    dates = sorted(set(dates))
    range_days = (dates[-1] - dates[0]).days + 1 if dates else 0
    return {
        "dated_files": len(dates),
        "range_days": range_days,
        "first_date": str(dates[0]) if dates else None,
        "last_date": str(dates[-1]) if dates else None,
    }


def generate_daily_status_report(
    project_root: Path,
    step_results: Dict[str, bool],
    feature_file: str | None,
    run_started_at: str,
) -> Dict[str, str]:
    """Generate daily status reports from real crawled files."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_iso = datetime.now().isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = project_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "danawa": project_root / "data" / "raw" / "danawa",
        "exchange": project_root / "data" / "raw" / "exchange",
        "news": project_root / "data" / "raw" / "news",
        "dataset": project_root / "data" / "processed" / "dataset",
    }

    details = {
        "danawa": _collect_date_summary(targets["danawa"]),
        "exchange": _collect_date_summary(targets["exchange"]),
        "news": _collect_date_summary(targets["news"]),
        "dataset": _collect_date_summary(targets["dataset"], prefix_to_strip="training_data_"),
    }

    today_files = {
        "danawa": str(targets["danawa"] / f"{today}.json"),
        "exchange": str(targets["exchange"] / f"{today}.json"),
        "news": str(targets["news"] / f"{today}.json"),
        "dataset": str(targets["dataset"] / f"training_data_{today}.json"),
    }
    today_exists = {k: Path(v).exists() for k, v in today_files.items()}

    ready_for_30d = min(v["range_days"] for v in details.values()) >= 30

    payload: Dict[str, Any] = {
        "generated_at": now_iso,
        "run_started_at": run_started_at,
        "date": today,
        "steps": step_results,
        "feature_file": feature_file,
        "today_files": today_files,
        "today_exists": today_exists,
        "coverage": details,
        "ready_for_30d_training": ready_for_30d,
    }

    json_path = reports_dir / f"data_status_{ts}.json"
    md_path = reports_dir / f"data_status_{ts}.md"
    latest_json = reports_dir / "latest_data_status.json"
    latest_md = reports_dir / "latest_data_status.md"

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    md_lines = [
        "# GPU Advisor 일일 데이터 상태 보고서",
        "",
        f"- 생성시각: {now_iso}",
        f"- 실행기준일: {today}",
        f"- 30일 학습 준비 여부: **{'ready' if ready_for_30d else 'collect_more_data'}**",
        "",
        "## 1) 실행 결과",
        f"- danawa: {'✓' if step_results.get('danawa') else '✗'}",
        f"- exchange: {'✓' if step_results.get('exchange') else '✗'}",
        f"- news: {'✓' if step_results.get('news') else '✗'}",
        f"- feature: {'✓' if step_results.get('feature') else '✗'}",
        f"- feature 파일: {feature_file or 'N/A'}",
        "",
        "## 2) 당일 파일 존재 여부",
        f"- danawa: {today_exists['danawa']}",
        f"- exchange: {today_exists['exchange']}",
        f"- news: {today_exists['news']}",
        f"- dataset: {today_exists['dataset']}",
        "",
        "## 3) 누적 커버리지",
    ]

    for key in ["danawa", "exchange", "news", "dataset"]:
        item = details[key]
        md_lines.extend(
            [
                f"### {key}",
                f"- dated_files: {item['dated_files']}",
                f"- range_days: {item['range_days']}",
                f"- first_date: {item['first_date']}",
                f"- last_date: {item['last_date']}",
                "",
            ]
        )

    md_text = "\n".join(md_lines).strip() + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
