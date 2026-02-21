from pathlib import Path

from backend.agent.release_pipeline import AgentReleasePipeline


def test_release_report_written_under_date_dir(tmp_path: Path):
    pipeline = AgentReleasePipeline(project_root=tmp_path)
    payload = {"status": "blocked", "readiness": {"target_days": 30, "current_min_days": 1, "remaining_days": 29}}

    reports = pipeline.write_report(payload)

    assert Path(reports["report_dir"]).exists()
    assert Path(reports["json_report"]).exists()
    assert Path(reports["markdown_report"]).exists()
    assert Path(reports["latest_json"]).exists()
    assert Path(reports["latest_markdown"]).exists()
    assert Path(reports["json_report"]).parent == Path(reports["report_dir"])
    assert Path(reports["markdown_report"]).parent == Path(reports["report_dir"])
