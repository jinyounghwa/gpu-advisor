import json
import logging
from fastapi import APIRouter, HTTPException
import numpy as np

from backend.agent import AgentEvaluator, AgentReleasePipeline, PipelineConfig, build_post_30d_next_steps, AgentConfig
from backend.agent.quality_gates import check_quality_gates
from backend.agent.sentiment import NewsSentimentAnalyzer
from backend.server.schemas import GPUQuery, PipelineRequest
from backend.server.dependencies import get_gpu_agent, get_repository, PROJECT_ROOT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

def _load_latest_news_snapshot() -> dict | None:
    news_dir = PROJECT_ROOT / "data" / "raw" / "news"
    if not news_dir.exists():
        return None
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return None
    return NewsSentimentAnalyzer().analyze_news_file(files[-1])

def _apply_realtime_news_to_state(state_vec: np.ndarray) -> tuple[np.ndarray, dict]:
    snapshot = _load_latest_news_snapshot()
    if snapshot is None:
        return state_vec, {"applied": False, "reason": "news_snapshot_not_found"}

    vec = np.asarray(state_vec, dtype=np.float32).copy()
    news_start = AgentConfig.news_start
    if len(vec) < news_start + 4:
        return vec, {"applied": False, "reason": "state_vector_too_short", "news": snapshot}

    vec[news_start + 0] = float((float(snapshot.get("sentiment_avg", 0.0)) + 1.0) / 2.0)
    vec[news_start + 1] = float(min(float(snapshot.get("total", 0)) / 100.0, 1.0))
    vec[news_start + 2] = float(min(float(snapshot.get("positive_count", 0)) / 50.0, 1.0))
    vec[news_start + 3] = float(min(float(snapshot.get("negative_count", 0)) / 50.0, 1.0))

    return vec, {"applied": True, "feature_start_index": news_start, "news": snapshot}

def _data_readiness() -> dict:
    readiness = AgentReleasePipeline(project_root=PROJECT_ROOT).check_readiness(target_days=30)
    readiness["ready_for_30d_training"] = readiness["ready_for_target"]
    return readiness

def _load_latest_release_result() -> dict | None:
    latest_report = PROJECT_ROOT / "docs" / "reports" / "latest_release_report.json"
    if not latest_report.exists():
        return None
    try:
        return json.loads(latest_report.read_text(encoding="utf-8"))
    except Exception:
        return None

@router.get("/readiness")
def agent_readiness():
    readiness = _data_readiness()
    status = "production_candidate" if readiness["ready_for_30d_training"] else "collect_more_data"
    return {"status": status, **readiness}

@router.get("/model-info")
def agent_model_info():
    return get_gpu_agent().get_model_info()

@router.get("/evaluate")
def agent_evaluate(lookback_days: int = 30):
    agent = get_gpu_agent()
    evaluator = AgentEvaluator(project_root=PROJECT_ROOT, agent=agent)
    return evaluator.run(lookback_days=lookback_days)

@router.get("/release-check")
def agent_release_check():
    readiness = _data_readiness()
    result = {"readiness": readiness}
    if not readiness["ready_for_30d_training"]:
        result["status"] = "blocked"
        result["reason"] = "insufficient_data_window"
        return result

    try:
        agent = get_gpu_agent()
        evaluator = AgentEvaluator(project_root=PROJECT_ROOT, agent=agent)
        metrics = evaluator.run(lookback_days=30)
    except Exception as e:
        result["status"] = "blocked"
        result["reason"] = f"evaluation_failed: {e}"
        return result

    gates = check_quality_gates(metrics)
    result["evaluation"] = metrics
    result["gates"] = gates
    result["status"] = "pass" if all(gates.values()) else "blocked"
    return result

@router.get("/next-steps")
def agent_next_steps():
    readiness = _data_readiness()
    normalized = {
        "target_days": readiness["target_days"],
        "current_min_days": readiness["current_min_days"],
        "remaining_days": readiness["remaining_days"],
        "ready_for_target": readiness["ready_for_30d_training"],
    }
    latest_release = _load_latest_release_result()
    return build_post_30d_next_steps(readiness=normalized, release_result=latest_release)

@router.post("/pipeline/run")
def run_agent_pipeline(req: PipelineRequest):
    pipeline = AgentReleasePipeline(project_root=PROJECT_ROOT)
    cfg = PipelineConfig(**req.model_dump())
    return pipeline.run(cfg)

@router.post("/ask")
def ask_gpu(query: GPUQuery):
    logger.info(f"Received query: {query.model_name}")
    if not query.model_name.strip():
        raise HTTPException(status_code=400, detail="모델명을 입력해주세요.")
    try:
        agent = get_gpu_agent()

        resolved = agent.resolve_state(query.model_name)
        if isinstance(resolved, tuple) and len(resolved) == 3:
            resolved_model, state_vec, data_date = resolved
            enriched_state_vec, news_context = _apply_realtime_news_to_state(state_vec)
            decision = agent.decide_from_state(resolved_model, enriched_state_vec, data_date)

            repo = get_repository()
            if news_context.get("applied"):
                repo.save_market_sentiment(news_context["news"])
            repo.save_agent_decision(
                query_model=query.model_name,
                resolved_model=resolved_model,
                data_date=data_date,
                decision=decision,
                news_context=news_context,
            )
        else:
            decision = agent.decide(query.model_name)
            news_context = {"applied": False, "reason": "legacy_agent_interface"}

        explanation = agent.explain(decision)
        action_probs_compact = ", ".join(f"{k}:{v * 100:.1f}%" for k, v in decision.action_probs.items())
        reward_compact = ", ".join(f"{k}:{v:.3f}" for k, v in decision.expected_rewards.items())

        return {
            "title": decision.gpu_model,
            "summary": f"AI Agent Decision: {decision.action}",
            "specs": f"Confidence {decision.confidence * 100:.1f}% | Value {decision.value:.3f}",
            "usage": f"MCTS {decision.simulations}회 계획 탐색 | 데이터 기준일 {decision.date}",
            "recommendation": explanation,
            "agent_trace": {
                "selected_action": decision.action,
                "raw_action": decision.raw_action,
                "confidence": decision.confidence,
                "entropy": decision.entropy,
                "value": decision.value,
                "safe_mode": decision.safe_mode,
                "safe_reason": decision.safe_reason,
                "action_probs": decision.action_probs,
                "expected_rewards": decision.expected_rewards,
                "action_probs_text": action_probs_compact,
                "expected_rewards_text": reward_compact,
                "news_context": news_context,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Agent inference failed")
        raise HTTPException(status_code=500, detail=f"AI agent error: {e}")
