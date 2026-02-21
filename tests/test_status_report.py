from pathlib import Path

from crawlers.status_report import generate_daily_status_report


def _write_json(path: Path, body: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_generate_daily_status_report(tmp_path: Path):
    root = tmp_path
    _write_json(root / "data/raw/danawa/2026-02-20.json")
    _write_json(root / "data/raw/danawa/2026-02-21.json")
    _write_json(root / "data/raw/exchange/2026-02-20.json")
    _write_json(root / "data/raw/exchange/2026-02-21.json")
    _write_json(root / "data/raw/news/2026-02-20.json")
    _write_json(root / "data/raw/news/2026-02-21.json")
    _write_json(root / "data/processed/dataset/training_data_2026-02-20.json")
    _write_json(root / "data/processed/dataset/training_data_2026-02-21.json")

    reports = generate_daily_status_report(
        project_root=root,
        step_results={"danawa": True, "exchange": True, "news": True, "feature": True},
        feature_file=str(root / "data/processed/dataset/training_data_2026-02-21.json"),
        run_started_at="2026-02-21T00:00:00",
    )

    assert Path(reports["json_report"]).exists()
    assert Path(reports["markdown_report"]).exists()
    assert Path(reports["latest_json"]).exists()
    assert Path(reports["latest_markdown"]).exists()

    latest_json_text = Path(reports["latest_json"]).read_text(encoding="utf-8")
    assert '"ready_for_30d_training": false' in latest_json_text
    assert '"range_days": 2' in latest_json_text
