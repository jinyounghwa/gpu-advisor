"""
AlphaZero API Routes
FastAPI endpoints for AlphaZero/MuZero training and inference
"""

import fastapi
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import torch
import numpy as np
import time
import psutil
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_alphazero_v2 import AlphaZeroTrainer
from inference.engine import InferenceEngine
from models.transformer_model import PolicyValueNetwork
from models.dynamics_network import DynamicsNetwork
from models.prediction_network import PredictionNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alphazero", tags=["alphazero"])

app = fastapi.FastAPI(title="AlphaZero API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingConfig(BaseModel):
    num_episodes: int = 100
    num_steps: int = 1000
    lr: float = 1e-4
    total_days: int = 200
    steps_per_day: int = 100


class InferenceRequest(BaseModel):
    state_vector: Optional[List[float]] = None
    use_mcts: bool = True
    use_kv_cache: bool = True


class GlobalState:
    trainer: Optional[AlphaZeroTrainer] = None
    inference_engine: Optional[InferenceEngine] = None
    is_training: bool = False
    training_metrics: List[Dict[str, Any]] = []
    latest_mcts_tree: Optional[Dict[str, Any]] = None
    latest_inference_result: Optional[Dict[str, Any]] = None

    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")


state = GlobalState()


@router.on_event("startup")
async def startup_event():
    logger.info("AlphaZero API server starting up...")


@router.on_event("shutdown")
async def shutdown_event():
    logger.info("AlphaZero API server shutting down...")


async def init_system():
    try:
        device_str = str(state.device)

        latent_dim = 256
        action_dim = 5

        policy_network = PolicyValueNetwork(
            input_dim=latent_dim,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=512,
            num_actions=action_dim,
            use_kv_cache=False,
        ).to(state.device)

        dynamics_network = DynamicsNetwork(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=512,
            num_layers=4,
        )

        prediction_network = PredictionNetwork(
            latent_dim=latent_dim,
            hidden_dim=512,
            num_layers=4,
            action_dim=action_dim,
        )

        config = {
            "lr": 1e-4,
            "mcts_simulations": 50,
            "mcts_exploration": 1.4142,
            "dirichlet_alpha": 0.03,
            "rollout_steps": 5,
        }

        state.trainer = AlphaZeroTrainer(
            policy_network,
            dynamics_network,
            prediction_network,
            config,
            device=device_str,
        )

        state.inference_engine = InferenceEngine(
            policy_network,
            dynamics_network,
            prediction_network,
            config,
            use_mcts=True,
            device=device_str,
        )

        return {"status": "initialized", "device": device_str}

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init")
async def init_system_api(config: TrainingConfig):
    try:
        await init_system()
        return {"status": "initialized", "config": config.dict()}

    except Exception as e:
        logger.error(f"Init error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/start")
async def start_training(background_tasks: BackgroundTasks):
    try:
        if state.is_training:
            return {"status": "already_running"}

        await init_system()

        num_episodes = 100
        num_steps = 1000
        lr = 1e-4
        total_days = 200
        steps_per_day = 100

        config = {
            "num_episodes": num_episodes,
            "num_steps": num_steps,
            "lr": lr,
            "total_days": total_days,
            "steps_per_day": steps_per_day,
        }

        async def training_loop(config_data: Dict):
            latent_dim = 256
            initial_states = np.random.randn(
                config_data["num_episodes"], latent_dim
            ).astype(np.float32)

            step = 0
            while state.is_training and step < config_data["num_steps"]:
                episodes_per_step = 2
                sampled_indices = np.random.choice(
                    len(initial_states), size=episodes_per_step, replace=True
                )
                sampled_states = initial_states[sampled_indices]

                training_data = state.trainer.generate_self_play_data(
                    episodes_per_step, sampled_states
                )

                if len(training_data) >= 32:
                    indices = np.random.choice(
                        len(training_data), size=32, replace=True
                    )
                    batch = [training_data[i] for i in indices]

                    metrics = state.trainer.train_step(batch)

                    training_metrics.append(
                        {
                            "type": "training_metrics",
                            "step": metrics["step"],
                            "episode": metrics.get("episode", 0),
                            "tps": metrics["tps"],
                            "vram_mb": metrics["vram_mb"],
                            "ram_mb": metrics["ram_mb"],
                            "loss": metrics["loss"],
                            "reward": metrics["reward"],
                            "timestamp": time.time(),
                        }
                    )

                    if len(training_data) > 0:
                        last_step = training_data[-1]
                        state.latest_mcts_tree = {
                            "root_value": float(last_step["value"]),
                            "action_probs": last_step["action_probs"].tolist(),
                            "top_actions": np.argsort(last_step["action_probs"])[::-1][
                                :3
                            ].tolist(),
                        }

                    step += 1

                    if step % 10 == 0:
                        logger.info(
                            f"Training step {step}/{config_data['num_steps']} completed"
                        )

            return training_metrics

        await background_tasks.add_task(training_loop, config)

        state.is_training = True
        logger.info(f"Training started: {num_steps} steps")

        return {"status": "started", "config": config, "device": str(state.device)}

    except Exception as e:
        logger.error(f"Start training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/stop")
async def stop_training():
    state.is_training = False
    logger.info("Training stopped")
    return {"status": "stopped"}


@router.get("/metrics/stream")
async def stream_metrics():
    async def event_generator():
        last_index = 0
        while True:
            if last_index < len(state.training_metrics):
                new_metrics = state.training_metrics[last_index:]
                for metric in new_metrics:
                    yield f"data: {json.dumps(metric)}\n\n"
                last_index += 1

            if not state.is_training and last_index == len(state.training_metrics):
                pass

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/inference/run")
async def run_inference(request: InferenceRequest):
    try:
        if state.inference_engine is None:
            await init_system()

        state_np = (
            np.array(request.state_vector)
            if request.state_vector
            else np.random.randn(256).astype(np.float32)
        )

        result = state.inference_engine.predict(
            state_np,
            use_kv_cache=request.use_kv_cache,
        )

        state.latest_inference_result = {
            "action": result["action"],
            "action_probs": result["action_probs"],
            "value": result["value"],
            "inference_time_ms": result["inference_time_ms"],
            "tree_depth": result["tree_depth"],
        }

        return result

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/quantize_benchmark")
async def quantize_benchmark():
    try:
        if state.inference_engine is None:
            await init_system()

        policy_net = state.inference_engine.policy_network
        if policy_net is None:
            raise HTTPException(
                status_code=500, detail="Inference engine not initialized"
            )

        original_device = next(policy_net.parameters()).device
        policy_net_cpu = policy_net.cpu()

        quantizer = ModelQuantizer()
        quantized_model = quantizer.quantize_dynamic(policy_net_cpu)

        input_tensor = torch.randn(1, 256).cpu()

        start = time.time()
        for _ in range(10):
            policy_net_cpu(input_tensor)
            fp32_latency = (time.time() - start) / 10 * 1000

        start = time.time()
        for _ in range(10):
            quantized_model(input_tensor)
            int8_latency = (time.time() - start) / 10 * 1000

        policy_net.to(original_device)

        size_fp32 = quantizer.get_model_size(policy_net_cpu)
        size_int8 = quantizer.get_model_size(quantized_model)

        return {
            "fp32_size_mb": size_fp32["total_size_mb"],
            "int8_size_mb": size_int8["total_size_mb"],
            "compression_ratio": size_fp32["total_size_bytes"]
            / size_int8["total_size_bytes"],
            "fp32_latency_ms": fp32_latency,
            "int8_latency_ms": int8_latency,
            "speedup": fp32_latency / int8_latency,
        }

    except Exception as e:
        logger.error(f"Quantization benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GPURequest(BaseModel):
    gpu_name: str


@router.post("/advisor/gpu")
async def advise_gpu(request: GPURequest):
    gpu_name = request.gpu_name.lower()
    recommendation = ""

    if "h100" in gpu_name:
        recommendation = "NVIDIA H100은 현존 최고의 AI 가속기입니다. 대규모 모델 학습과 추론에 최적화되어 있습니다. 다만 가격이 매우 높으므로 엔터프라이즈 용도에 적합합니다."
    elif "a100" in gpu_name:
        recommendation = "NVIDIA A100은 여전히 강력한 성능을 보여주는 데이터센터용 GPU입니다. H100보다는 느리지만 대부분의 AI 작업에 충분합니다."
    elif "4090" in gpu_name:
        recommendation = "RTX 4090은 소비자용 GPU 중 최고의 성능을 자랑합니다. 24GB VRAM은 웬만한 LLM 파인튜닝이나 7B~13B 모델 추론에 매우 적합합니다."
    elif "4080" in gpu_name or "4070" in gpu_name:
        recommendation = "RTX 4080/4070 라인업은 가성비 좋은 선택입니다. 추론용으로는 훌륭하지만, VRAM 용량에 따라 학습 시 배치 크기에 제약이 있을 수 있습니다."
    elif "3090" in gpu_name:
        recommendation = "RTX 3090은 24GB VRAM을 가진 훌륭한 중고 가성비 픽입니다. 4090보다 느리지만 메모리 용량 덕분에 여전히 현역입니다."
    elif "m4" in gpu_name or "apple" in gpu_name or "mac" in gpu_name:
        recommendation = "Apple M4 칩셋은 통합 메모리 구조 덕분에 대용량 모델을 올리기에 유리합니다. MPS 가속을 통해 PyTorch 학습도 가능하지만, NVIDIA GPU보다는 속도가 느릴 수 있습니다. (현재 시스템이 이 환경에서 구동 중일 수 있습니다)"
    else:
        recommendation = f"'{request.gpu_name}'에 대한 구체적인 정보는 없지만, VRAM 용량이 16GB 이상이라면 AI 학습에, 8GB 이상이라면 가벼운 추론에 적합할 것입니다."

    return {"gpu_name": request.gpu_name, "recommendation": recommendation}


@router.get("/system/status")
async def get_system_status():
    process = psutil.Process()

    try:
        memory_info = process.memory_info()

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_mb": memory_info.rss / 1024 / 1024,
            "is_training": state.is_training,
            "device": str(state.trainer.device) if state.trainer else "unknown",
        }

    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
