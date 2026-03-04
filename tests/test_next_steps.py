from backend.agent.next_steps import build_post_30d_next_steps


def test_next_steps_before_30_days():
    plan = build_post_30d_next_steps(
        readiness={
            "target_days": 30,
            "current_min_days": 12,
            "remaining_days": 18,
            "ready_for_target": False,
        }
    )

    assert plan["ready_for_target"] is False
    assert plan["remaining_days"] == 18
    assert plan["steps"][0]["id"] == "collect_daily_data"
    assert plan["steps"][0]["status"] == "in_progress"
    assert plan["steps"][-1]["id"] == "wait_for_30d_unlock"


def test_next_steps_after_30_days_with_pass():
    plan = build_post_30d_next_steps(
        readiness={
            "target_days": 30,
            "current_min_days": 30,
            "remaining_days": 0,
            "ready_for_target": True,
        },
        release_result={"status": "pass"},
    )

    status_by_id = {step["id"]: step["status"] for step in plan["steps"]}
    assert status_by_id["run_release_pipeline"] == "completed"
    assert status_by_id["review_release_report"] == "completed"
    assert status_by_id["tag_release_candidate"] == "ready"
