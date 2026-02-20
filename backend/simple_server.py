from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import json
import time
import psutil
import os
from typing import Optional
from pathlib import Path
from datetime import datetime

try:
    from agent import (
        GPUPurchaseAgent,
        AgentFineTuner,
        AgentEvaluator,
        AgentReleasePipeline,
        PipelineConfig,
    )
except ModuleNotFoundError:  # pragma: no cover
    from backend.agent import (
        GPUPurchaseAgent,
        AgentFineTuner,
        AgentEvaluator,
        AgentReleasePipeline,
        PipelineConfig,
    )

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU Advisor + AI Training Dashboard")

# CORS 설정 (프론트엔드 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GPUQuery(BaseModel):
    model_name: str


class TrainingConfig(BaseModel):
    num_steps: int = 500
    learning_rate: float = 1e-4
    batch_size: int = 32
    seed: int = 42


class PipelineRequest(BaseModel):
    target_days: int = 30
    lookback_days: int = 30
    num_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    min_accuracy: float = 0.55
    min_avg_reward: float = 0.0
    max_abstain_ratio: float = 0.85
    max_safe_override_ratio: float = 0.90
    min_action_entropy: float = 0.25
    min_uplift_vs_buy: float = 0.0
    require_30d: bool = True
    run_training: bool = True


# Training state
class TrainingState:
    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.metrics_history = []
        self.start_time = None
        self.config = None

    def reset(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.metrics_history = []
        self.start_time = None
        self.config = None


training_state = TrainingState()
gpu_agent: Optional[GPUPurchaseAgent] = None


def get_gpu_agent() -> GPUPurchaseAgent:
    global gpu_agent
    if gpu_agent is None:
        gpu_agent = GPUPurchaseAgent()
    return gpu_agent


def _data_readiness() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    targets = {
        "danawa": project_root / "data" / "raw" / "danawa",
        "exchange": project_root / "data" / "raw" / "exchange",
        "news": project_root / "data" / "raw" / "news",
        "dataset": project_root / "data" / "processed" / "dataset",
    }
    details = {}
    min_days = None

    for name, root in targets.items():
        dates = []
        for f in sorted(root.glob("*.json")):
            stem = f.stem.replace("training_data_", "")
            try:
                dates.append(datetime.strptime(stem, "%Y-%m-%d").date())
            except ValueError:
                continue
        dates = sorted(set(dates))
        range_days = (dates[-1] - dates[0]).days + 1 if dates else 0
        details[name] = {
            "dated_files": len(dates),
            "range_days": range_days,
            "first_date": str(dates[0]) if dates else None,
            "last_date": str(dates[-1]) if dates else None,
        }
        if min_days is None:
            min_days = range_days
        else:
            min_days = min(min_days, range_days)

    min_days = min_days or 0
    target_days = 30
    return {
        "target_days": target_days,
        "current_min_days": min_days,
        "remaining_days": max(target_days - min_days, 0),
        "ready_for_30d_training": min_days >= target_days,
        "details": details,
    }


# ===== GPU Advisor Endpoints =====

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.get("/api/agent/readiness")
def agent_readiness():
    readiness = _data_readiness()
    status = (
        "production_candidate"
        if readiness["ready_for_30d_training"]
        else "collect_more_data"
    )
    return {"status": status, **readiness}


@app.get("/api/agent/model-info")
def agent_model_info():
    agent = get_gpu_agent()
    return agent.get_model_info()


@app.get("/api/agent/evaluate")
def agent_evaluate(lookback_days: int = 30):
    project_root = Path(__file__).resolve().parents[1]
    agent = get_gpu_agent()
    evaluator = AgentEvaluator(project_root=project_root, agent=agent)
    return evaluator.run(lookback_days=lookback_days)


@app.get("/api/agent/release-check")
def agent_release_check():
    readiness = _data_readiness()
    result = {"readiness": readiness}
    if not readiness["ready_for_30d_training"]:
        result["status"] = "blocked"
        result["reason"] = "insufficient_data_window"
        return result

    try:
        metrics = agent_evaluate(lookback_days=30)
    except Exception as e:
        result["status"] = "blocked"
        result["reason"] = f"evaluation_failed: {e}"
        return result

    gates = {
        "accuracy_raw": metrics["directional_accuracy_buy_vs_wait_raw"] >= 0.55,
        "reward_raw": metrics["avg_reward_per_decision_raw"] > 0.0,
        "abstain": metrics["abstain_ratio"] <= 0.85,
        "safe_override": metrics.get("safe_override_ratio", 1.0) <= 0.90,
        "action_entropy_raw": metrics.get("action_entropy_raw", 0.0) >= 0.25,
        "uplift_raw_vs_buy": metrics.get("uplift_raw_vs_always_buy", -1e9) >= 0.0,
        "no_mode_collapse_raw": not bool(metrics.get("mode_collapse_raw", True)),
    }
    result["evaluation"] = metrics
    result["gates"] = gates
    result["status"] = "pass" if all(gates.values()) else "blocked"
    return result


@app.post("/api/agent/pipeline/run")
def run_agent_pipeline(req: PipelineRequest):
    project_root = Path(__file__).resolve().parents[1]
    pipeline = AgentReleasePipeline(project_root=project_root)
    cfg = PipelineConfig(
        target_days=req.target_days,
        lookback_days=req.lookback_days,
        num_steps=req.num_steps,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        seed=req.seed,
        min_accuracy=req.min_accuracy,
        min_avg_reward=req.min_avg_reward,
        max_abstain_ratio=req.max_abstain_ratio,
        max_safe_override_ratio=req.max_safe_override_ratio,
        min_action_entropy=req.min_action_entropy,
        min_uplift_vs_buy=req.min_uplift_vs_buy,
        require_30d=req.require_30d,
        run_training=req.run_training,
    )
    return pipeline.run(cfg)


@app.post("/api/ask")
def ask_gpu(query: GPUQuery):
    logger.info(f"Received query: {query.model_name}")
    if not query.model_name.strip():
        raise HTTPException(status_code=400, detail="모델명을 입력해주세요.")
    try:
        agent = get_gpu_agent()
        decision = agent.decide(query.model_name)
        explanation = agent.explain(decision)

        action_probs_compact = ", ".join(
            f"{k}:{v * 100:.1f}%" for k, v in decision.action_probs.items()
        )
        reward_compact = ", ".join(
            f"{k}:{v:.3f}" for k, v in decision.expected_rewards.items()
        )

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
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Agent inference failed")
        raise HTTPException(status_code=500, detail=f"AI agent error: {e}")


# ===== Training Endpoints =====

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """학습 시작"""
    if training_state.is_training:
        raise HTTPException(status_code=400, detail="Training already running")

    training_state.reset()
    training_state.is_training = True
    training_state.total_steps = config.num_steps
    training_state.start_time = time.time()
    training_state.config = config

    # Background training loop (real data fine-tuning)
    async def training_loop():
        global gpu_agent
        try:
            project_root = Path(__file__).resolve().parents[1]
            trainer = AgentFineTuner(project_root=project_root)

            def on_step(metric: dict) -> None:
                training_state.metrics_history.append(metric)
                training_state.current_step = metric["step"]

            def should_stop() -> bool:
                return not training_state.is_training

            await asyncio.to_thread(
                trainer.run,
                config.num_steps,
                config.batch_size,
                config.learning_rate,
                on_step,
                should_stop,
                config.seed,
            )
            gpu_agent = None
        except Exception as e:
            logger.exception("Training loop failed")
            training_state.metrics_history.append(
                {"type": "error", "message": str(e), "timestamp": time.time()}
            )
        finally:
            training_state.is_training = False

    asyncio.create_task(training_loop())

    return {
        "status": "started",
        "config": config.dict(),
        "total_steps": config.num_steps,
    }


@app.post("/api/training/stop")
async def stop_training():
    """학습 중지"""
    training_state.is_training = False
    return {"status": "stopped", "steps_completed": training_state.current_step}


@app.get("/api/training/status")
async def training_status():
    """현재 학습 상태"""
    latest = training_state.metrics_history[-1] if training_state.metrics_history else None
    return {
        "is_training": training_state.is_training,
        "current_step": training_state.current_step,
        "total_steps": training_state.total_steps,
        "latest_metrics": latest,
        "history_length": len(training_state.metrics_history),
    }


@app.get("/api/training/metrics")
async def get_metrics(last_n: int = 50):
    """최근 N개 메트릭 반환"""
    history = training_state.metrics_history
    if last_n > 0:
        history = history[-last_n:]
    return {"metrics": history, "total": len(training_state.metrics_history)}


@app.get("/api/training/metrics/stream")
async def stream_metrics():
    """SSE 스트리밍 메트릭"""
    async def event_generator():
        last_index = 0
        while True:
            if last_index < len(training_state.metrics_history):
                new_metrics = training_state.metrics_history[last_index:]
                for metric in new_metrics:
                    yield f"data: {json.dumps(metric)}\n\n"
                last_index = len(training_state.metrics_history)

            if not training_state.is_training and last_index >= len(training_state.metrics_history):
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

            await asyncio.sleep(0.2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/training/summary")
async def training_summary():
    """학습 결과 요약"""
    if not training_state.metrics_history:
        return {"message": "No training data available"}

    history = [m for m in training_state.metrics_history if isinstance(m, dict) and "loss" in m]
    if not history:
        return {"message": "No numeric training metrics available yet"}
    recent = history[-50:] if len(history) > 50 else history

    return {
        "total_steps": len(history),
        "avg_loss": round(sum(m["loss"] for m in recent) / len(recent), 4),
        "min_loss": round(min(m["loss"] for m in history), 4),
        "avg_reward": round(sum(m["reward"] for m in recent) / len(recent), 4),
        "max_reward": round(max(m["reward"] for m in history), 4),
        "avg_tps": round(sum(m["tps"] for m in recent) / len(recent), 1),
        "avg_vram_mb": round(sum(m["vram_mb"] for m in recent) / len(recent), 1),
        "avg_win_rate": round(sum(m["win_rate"] for m in recent) / len(recent), 4),
        "total_episodes": history[-1]["episode"] if history else 0,
        "elapsed_time": round(history[-1]["elapsed_time"], 1) if history else 0,
    }


@app.get("/api/system/status")
async def system_status():
    """시스템 상태"""
    process = psutil.Process(os.getpid())
    memory = process.memory_info()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "memory_mb": round(memory.rss / (1024 * 1024), 1),
        "memory_percent": round(psutil.virtual_memory().percent, 1),
        "disk_percent": round(psutil.disk_usage("/").percent, 1),
        "is_training": training_state.is_training,
        "uptime": round(time.time() - (training_state.start_time or time.time()), 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
