#!/usr/bin/env python3
"""
Run full agent release pipeline from CLI.
"""

from pathlib import Path
from backend.agent import AgentReleasePipeline, PipelineConfig


def main():
    root = Path(__file__).resolve().parents[1]
    pipeline = AgentReleasePipeline(project_root=root)
    cfg = PipelineConfig()
    result = pipeline.run(cfg)
    print(result)


if __name__ == "__main__":
    main()

