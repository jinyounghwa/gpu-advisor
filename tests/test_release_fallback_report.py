from pathlib import Path

from crawlers.release_report_fallback import write_fallback_release_report


def test_write_fallback_release_report(tmp_path: Path):
    reports = write_fallback_release_report(
        project_root=tmp_path,
        reason="release_pipeline_failed",
        detail="subprocess crashed",
        run_started_at="2026-02-24T00:00:01",
    )

    assert Path(reports["report_dir"]).exists()
    assert Path(reports["json_report"]).exists()
    assert Path(reports["markdown_report"]).exists()
    assert Path(reports["latest_json"]).exists()
    assert Path(reports["latest_markdown"]).exists()

    json_text = Path(reports["json_report"]).read_text(encoding="utf-8")
    assert '"status": "blocked"' in json_text
    assert '"fallback_report": true' in json_text
