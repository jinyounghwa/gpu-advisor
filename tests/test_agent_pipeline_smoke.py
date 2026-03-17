from pathlib import Path

import pytest

from backend.agent import (
    AgentEvaluator,
    AgentReleasePipeline,
    GPUPurchaseAgent,
    PipelineConfig,
)


def test_agent_pipeline_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "data" / "processed" / "dataset"
    has_dataset = any(dataset_dir.glob("training_data*.json"))
    if not has_dataset:
        pytest.skip("No processed dataset found; skipping smoke test")

    agent = GPUPurchaseAgent(project_root=root)
    info = agent.get_model_info()
    assert "checkpoint_path" in info
    assert "meta" in info

    decision = agent.decide("RTX 4090")
    assert decision.action in {"BUY_NOW", "WAIT_SHORT", "WAIT_LONG", "HOLD", "SKIP"}
    assert 0.0 <= decision.confidence <= 1.0

    evaluator = AgentEvaluator(project_root=root, agent=agent)
    metrics = evaluator.run(lookback_days=7)
    for key in [
        "samples",
        "avg_reward_per_decision",
        "directional_accuracy_buy_vs_wait",
        "abstain_ratio",
        "action_entropy",
        "uplift_vs_always_buy",
    ]:
        assert key in metrics, f"missing metric: {key}"

    pipeline = AgentReleasePipeline(project_root=root)
    result = pipeline.run(
        PipelineConfig(
            require_30d=False,
            run_training=False,
            lookback_days=7,
        )
    )
    assert "status" in result
    assert "reports" in result
