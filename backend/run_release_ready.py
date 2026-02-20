#!/usr/bin/env python3
"""
One-click release runner for 30-day-ready workflow.
Stages: train -> evaluate -> release gates -> optional git tag/push.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent import AgentReleasePipeline, PipelineConfig


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def create_and_push_tag(tag: str, push: bool) -> dict:
    rc, out = run_cmd(["git", "tag", tag])
    if rc != 0:
        return {"ok": False, "step": "tag", "message": out}
    if not push:
        return {"ok": True, "step": "tag", "message": f"created local tag: {tag}"}
    rc, out = run_cmd(["git", "push", "origin", tag])
    if rc != 0:
        return {"ok": False, "step": "push-tag", "message": out}
    return {"ok": True, "step": "push-tag", "message": f"pushed tag: {tag}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-click release runner")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--allow-short-window", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--tag", action="store_true")
    parser.add_argument("--push-tag", action="store_true")
    parser.add_argument("--tag-prefix", default="release-agent")
    args = parser.parse_args()

    project_root = ROOT
    pipeline = AgentReleasePipeline(project_root=project_root)

    cfg = PipelineConfig(
        target_days=30,
        lookback_days=args.lookback_days,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        require_30d=not args.allow_short_window,
        run_training=not args.no_train,
    )

    result = pipeline.run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("status") != "pass":
        print("\n[release-ready] 판정: blocked")
        return 1

    if args.tag:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"{args.tag_prefix}-{ts}"
        tag_result = create_and_push_tag(tag, args.push_tag)
        print("\n[release-ready] tag result")
        print(json.dumps(tag_result, ensure_ascii=False, indent=2))
        return 0 if tag_result.get("ok") else 2

    print("\n[release-ready] 판정: pass (tag 미생성)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
