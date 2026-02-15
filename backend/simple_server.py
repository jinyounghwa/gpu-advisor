from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import json
import time
import math
import random
import psutil
import os

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


def simulate_training_step(step: int, total_steps: int) -> dict:
    """Simulate realistic training metrics"""
    progress = step / max(total_steps, 1)

    # Loss: starts high, decreases with noise
    base_loss = 2.5 * math.exp(-3.0 * progress) + 0.15
    noise = random.gauss(0, 0.08 * (1 - progress * 0.5))
    loss = max(0.01, base_loss + noise)

    # Reward: starts low, increases
    base_reward = -1.0 + 3.5 * (1 - math.exp(-2.5 * progress))
    reward_noise = random.gauss(0, 0.15)
    reward = base_reward + reward_noise

    # TPS: fluctuates around a base value
    base_tps = 145 + random.gauss(0, 15)
    tps = max(50, base_tps)

    # VRAM usage: increases slightly over training
    vram_base = 1200 + progress * 800
    vram_mb = vram_base + random.gauss(0, 50)

    # RAM usage
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    # Policy loss and value loss components
    policy_loss = loss * 0.6 + random.gauss(0, 0.02)
    value_loss = loss * 0.4 + random.gauss(0, 0.01)

    # Entropy: decreases as policy becomes more certain
    entropy = 1.5 * math.exp(-1.5 * progress) + 0.1 + random.gauss(0, 0.05)

    # Learning rate with warmup and decay
    if progress < 0.1:
        lr = 1e-4 * (progress / 0.1)
    else:
        lr = 1e-4 * math.cos(math.pi / 2 * (progress - 0.1) / 0.9)

    # Gradient norm
    grad_norm = 2.0 * math.exp(-progress) + 0.5 + random.gauss(0, 0.3)
    grad_norm = max(0.1, grad_norm)

    # Win rate: increases over time
    win_rate = 0.3 + 0.5 * (1 - math.exp(-3.0 * progress)) + random.gauss(0, 0.03)
    win_rate = max(0, min(1, win_rate))

    # Episode length
    episode_length = int(50 + 150 * progress + random.gauss(0, 10))
    episode_length = max(10, episode_length)

    # Action distribution (5 actions)
    action_probs = [random.random() for _ in range(5)]
    total = sum(action_probs)
    # Make action distribution more peaked as training progresses
    temperature = max(0.3, 1.0 - progress * 0.7)
    action_probs = [p ** (1.0 / temperature) for p in action_probs]
    total = sum(action_probs)
    action_probs = [p / total for p in action_probs]

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=None)

    elapsed = time.time() - training_state.start_time if training_state.start_time else 0

    return {
        "step": step,
        "episode": step // 50,
        "timestamp": time.time(),
        "elapsed_time": elapsed,
        # Core metrics
        "loss": round(loss, 4),
        "policy_loss": round(max(0.01, policy_loss), 4),
        "value_loss": round(max(0.01, value_loss), 4),
        "reward": round(reward, 4),
        "entropy": round(max(0.01, entropy), 4),
        # Performance
        "tps": round(tps, 1),
        "vram_mb": round(vram_mb, 1),
        "ram_mb": round(ram_mb, 1),
        "cpu_percent": round(cpu_percent, 1),
        # Training details
        "learning_rate": round(lr, 8),
        "grad_norm": round(grad_norm, 4),
        "win_rate": round(win_rate, 4),
        "episode_length": episode_length,
        # Action distribution
        "action_probs": [round(p, 4) for p in action_probs],
        # Progress
        "progress": round(progress * 100, 1),
    }


# ===== GPU Advisor Endpoints =====

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.post("/api/ask")
def ask_gpu(query: GPUQuery):
    logger.info(f"Received query: {query.model_name}")
    name = query.model_name.lower().strip()

    if not name:
        raise HTTPException(status_code=400, detail="모델명을 입력해주세요.")

    if "4090" in name:
        return {
            "title": "NVIDIA GeForce RTX 4090",
            "summary": "현존 소비자용 최강의 GPU입니다.",
            "specs": "24GB VRAM, 16384 CUDA Cores",
            "usage": "대규모 AI 학습, 4K 게이밍, 3D 렌더링",
            "recommendation": "가격이 문제되지 않는다면 최고의 선택입니다. LLM 로컬 구동에 최적입니다.",
        }
    elif "4080" in name:
        return {
            "title": "NVIDIA GeForce RTX 4080",
            "summary": "하이엔드급 성능을 보여줍니다.",
            "specs": "16GB VRAM",
            "usage": "고사양 게이밍, 중규모 AI 작업",
            "recommendation": "4090이 부담스럽지만 고성능이 필요할 때 좋습니다.",
        }
    elif "h100" in name:
        return {
            "title": "NVIDIA H100 Tensor Core",
            "summary": "데이터센터/엔터프라이즈 전용 괴물 칩셋입니다.",
            "specs": "80GB HBM3",
            "usage": "초대규모 AI 모델 학습 (ChatGPT급)",
            "recommendation": "개인용이 아닙니다. 서버실이나 클라우드 환경에 적합합니다.",
        }
    elif "m4" in name or "mac" in name:
        return {
            "title": "Apple M4 Chip",
            "summary": "Apple Silicon의 최신 프로세서입니다.",
            "specs": "Unified Memory Architecture",
            "usage": "온디바이스 AI, 영상 편집, 효율적인 작업",
            "recommendation": "전성비가 매우 뛰어나며, PyTorch MPS 가속을 통해 학습도 가능합니다.",
        }
    else:
        return {
            "title": f"Unknown GPU: {query.model_name}",
            "summary": "정보를 찾을 수 없습니다.",
            "specs": "-",
            "usage": "-",
            "recommendation": "정확한 모델명을 입력해 주세요 (예: RTX 4090, H100)",
        }


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

    # Background training loop
    async def training_loop():
        for step in range(config.num_steps):
            if not training_state.is_training:
                break
            metrics = simulate_training_step(step, config.num_steps)
            training_state.metrics_history.append(metrics)
            training_state.current_step = step + 1
            await asyncio.sleep(0.1)  # ~10 steps/sec simulation speed

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

    history = training_state.metrics_history
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
