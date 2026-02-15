"""
FastAPI 서버 - 학습 제어 및 상태 조회
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.gym_env import DummyMarketEnv
from models.transformer_model import create_model
from api.training import Trainer, TrainingMetrics
from api.alphazero_routes import router as alphazero_router


app = FastAPI(title="RL Training Benchmark API")

app.include_router(alphazero_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingConfig(BaseModel):
    total_days: int = 200
    steps_per_day: int = 100
    learning_rate: float = 1e-4
    num_steps: int = 1000


class TrainingStatus(BaseModel):
    status: str  # "idle", "running", "completed"
    current_step: int
    total_steps: int
    tps: float
    vram_mb: float
    ram_mb: float
    loss: float
    reward: float
    elapsed_time: float
    predicted_total_time: float


is_training = False


def get_data_path() -> str:
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return os.path.join(
        project_root, "data", "processed", "dataset", "training_data.json"
    )


@app.get("/")
async def root():
    is_training = getattr(app.state, "is_training", False)

    return {
        "message": "RL Training Benchmark API",
        "status": "running" if is_training else "idle",
    }


@app.post("/api/init")
async def init_training(config: TrainingConfig):
    """학습 환경 초기화"""
    is_training = getattr(app.state, "is_training", False)

    if is_training:
        raise HTTPException(status_code=400, detail="Training is already running")

    if hasattr(app.state, "trainer") and app.state.trainer is not None:
        raise HTTPException(status_code=400, detail="Training already initialized")

    try:
        data_path = get_data_path()

        env_config = {"data_path": data_path, "total_days": config.total_days}

        env = DummyMarketEnv(**env_config)

        model_config = {
            "input_dim": 11,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 1024,
            "num_actions": 2,
        }

        model = create_model(model_config)

        trainer_config = {
            "learning_rate": config.learning_rate,
            "total_steps": config.total_days * config.steps_per_day,
        }

        trainer = Trainer(model, env, trainer_config)

        app.state.trainer = trainer
        app.state.env = env

        return {
            "message": "Training environment initialized",
            "model_params": model.count_parameters(),
            "total_steps": trainer_config["total_steps"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """현재 학습 상태 조회"""
    is_training = getattr(app.state, "is_training", False)
    trainer = getattr(app.state, "trainer", None)

    if trainer is None:
        return TrainingStatus(
            status="idle",
            current_step=0,
            total_steps=0,
            tps=0.0,
            vram_mb=0.0,
            ram_mb=0.0,
            loss=0.0,
            reward=0.0,
            elapsed_time=0.0,
            predicted_total_time=0.0,
        )

    if not trainer.metrics_history:
        return TrainingStatus(
            status="ready",
            current_step=0,
            total_steps=trainer.config["total_steps"],
            tps=0.0,
            vram_mb=0.0,
            ram_mb=0.0,
            loss=0.0,
            reward=0.0,
            elapsed_time=0.0,
            predicted_total_time=0.0,
        )

    latest = trainer.metrics_history[-1]

    is_training_status = getattr(app.state, "is_training", False)

    return TrainingStatus(
        status="running" if is_training_status else "paused",
        current_step=latest.step,
        total_steps=trainer.config["total_steps"],
        tps=latest.tps,
        vram_mb=latest.vram_mb,
        ram_mb=latest.ram_mb,
        loss=latest.loss,
        reward=latest.reward,
        elapsed_time=latest.elapsed_time,
        predicted_total_time=latest.predicted_total_time,
    )


@app.post("/api/start")
async def start_training():
    """학습 시작"""
    global is_training

    trainer = getattr(app.state, "trainer", None)

    if trainer is None:
        raise HTTPException(
            status_code=400, detail="Training environment not initialized"
        )

    if is_training:
        raise HTTPException(status_code=400, detail="Training is already running")

    is_training = True
    app.state.is_training = True

    return {"message": "Training started"}

    is_training = True
    return {"message": "Training started"}


@app.post("/api/stop")
async def stop_training():
    """학습 중지"""
    global is_training

    is_training = False
    app.state.is_training = False

    return {"message": "Training stopped"}


@app.get("/api/metrics/stream")
async def stream_metrics():
    """학습 메트릭 실시간 스트리밍"""

    async def event_generator():
        last_sent_step = -1

        while True:
            trainer = getattr(app.state, "trainer", None)
            is_training = getattr(app.state, "is_training", False)

            if trainer is None:
                await asyncio.sleep(1)
                continue

            if is_training:
                try:
                    metrics = trainer.train_step(last_sent_step + 1)

                    data = {
                        "step": metrics.step,
                        "episode": metrics.episode,
                        "tps": metrics.tps,
                        "vram_mb": metrics.vram_mb,
                        "ram_mb": metrics.ram_mb,
                        "loss": metrics.loss,
                        "reward": metrics.reward,
                        "elapsed_time": metrics.elapsed_time,
                        "predicted_total_time": metrics.predicted_total_time,
                    }

                    yield f"data: {json.dumps(data)}\n\n"
                    last_sent_step = metrics.step

                except Exception as e:
                    error_data = {"error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    app.state.is_training = False

            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """학습 메트릭 요약"""
    trainer = getattr(app.state, "trainer", None)

    if trainer is None or not trainer.metrics_history:
        return {
            "count": 0,
            "avg_tps": 0.0,
            "max_tps": 0.0,
            "avg_vram_mb": 0.0,
            "max_vram_mb": 0.0,
            "avg_loss": 0.0,
        }

    metrics_list = trainer.metrics_history

    return {
        "count": len(metrics_list),
        "avg_tps": sum(m.tps for m in metrics_list) / len(metrics_list),
        "max_tps": max(m.tps for m in metrics_list),
        "min_tps": min(m.tps for m in metrics_list),
        "avg_vram_mb": sum(m.vram_mb for m in metrics_list) / len(metrics_list),
        "max_vram_mb": max(m.vram_mb for m in metrics_list),
        "avg_ram_mb": sum(m.ram_mb for m in metrics_list) / len(metrics_list),
        "avg_loss": sum(m.loss for m in metrics_list) / len(metrics_list),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
