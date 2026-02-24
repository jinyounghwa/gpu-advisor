#!/usr/bin/env python3
"""
Run daily release check and print machine-readable JSON result.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent import AgentReleasePipeline, PipelineConfig


def main() -> int:
    try:
        pipeline = AgentReleasePipeline(project_root=ROOT)
        cfg = PipelineConfig(
            run_training=False,
            require_30d=False,
            lookback_days=30,
        )
        result = pipeline.run(cfg)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
