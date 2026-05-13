import asyncio
import time
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.agent import AgentFineTuner
from backend.server.schemas import TrainingConfig
from backend.server.state import training_state
from backend.server.dependencies import set_gpu_agent, PROJECT_ROOT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

@router.post("/start")
async def start_training(config: TrainingConfig):
    if training_state.snapshot()["is_training"]:
        raise HTTPException(status_code=400, detail="Training already running")

    training_state.reset()
    training_state.configure(
        total_steps=config.num_steps,
        start_time=time.time(),
        config=config,
    )

    async def training_loop():
        try:
            trainer = AgentFineTuner(project_root=PROJECT_ROOT)

            def on_step(metric: dict) -> None:
                training_state.append_metric(metric)

            def should_stop() -> bool:
                return training_state.should_stop()

            await asyncio.to_thread(
                trainer.run,
                config.num_steps,
                config.batch_size,
                config.learning_rate,
                on_step,
                should_stop,
                config.seed,
            )
            set_gpu_agent(None) # reset agent
        except Exception as e:
            logger.exception("Training loop failed")
            training_state.append_metric(
                {"type": "error", "message": str(e), "timestamp": time.time()}
            )
        finally:
            training_state.stop()

    asyncio.create_task(training_loop())

    return {
        "status": "started",
        "config": config.model_dump(),
        "total_steps": config.num_steps,
    }

@router.post("/stop")
async def stop_training():
    snapshot = training_state.snapshot()
    training_state.stop()
    return {"status": "stopped", "steps_completed": snapshot["current_step"]}

@router.get("/status")
async def training_status():
    snapshot = training_state.snapshot()
    latest = snapshot["metrics_history"][-1] if snapshot["metrics_history"] else None
    return {
        "is_training": snapshot["is_training"],
        "current_step": snapshot["current_step"],
        "total_steps": snapshot["total_steps"],
        "latest_metrics": latest,
        "history_length": len(snapshot["metrics_history"]),
    }

@router.get("/metrics")
async def get_metrics(last_n: int = 50):
    snapshot = training_state.snapshot()
    history = snapshot["metrics_history"]
    if last_n > 0:
        history = history[-last_n:]
    return {"metrics": history, "total": len(snapshot["metrics_history"])}

@router.get("/metrics/stream")
async def stream_metrics():
    async def event_generator():
        last_index = 0
        while True:
            snapshot = training_state.snapshot()
            if last_index < len(snapshot["metrics_history"]):
                new_metrics = snapshot["metrics_history"][last_index:]
                for metric in new_metrics:
                    yield f"data: {json.dumps(metric)}\n\n"
                last_index = len(snapshot["metrics_history"])

            if not snapshot["is_training"] and last_index >= len(snapshot["metrics_history"]):
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

            await asyncio.sleep(0.2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/summary")
async def training_summary():
    snapshot = training_state.snapshot()
    if not snapshot["metrics_history"]:
        return {"message": "No training data available"}

    history = [m for m in snapshot["metrics_history"] if isinstance(m, dict) and "loss" in m]
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
        "total_episodes": history[-1].get("episode", 0) if history else 0,
        "elapsed_time": round(history[-1].get("elapsed_time", 0), 1) if history else 0,
    }
