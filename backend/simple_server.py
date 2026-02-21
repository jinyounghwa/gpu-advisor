from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

try:
    from agent import (
        AgentEvaluator,
        AgentFineTuner,
        AgentReleasePipeline,
        GPUPurchaseAgent,
        PipelineConfig,
    )
except ModuleNotFoundError:  # pragma: no cover
    from backend.agent import (
        AgentEvaluator,
        AgentFineTuner,
        AgentReleasePipeline,
        GPUPurchaseAgent,
        PipelineConfig,
    )

try:
    from storage import RepositoryProtocol, create_repository
except ModuleNotFoundError:  # pragma: no cover
    from backend.storage import RepositoryProtocol, create_repository

try:
    from api.sentiment import NewsSentimentAnalyzer
except ModuleNotFoundError:  # pragma: no cover
    from backend.api.sentiment import NewsSentimentAnalyzer

try:
    from security import AuthMode, SecurityConfig
except ModuleNotFoundError:  # pragma: no cover
    from backend.security import AuthMode, SecurityConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU Advisor + AI Training Dashboard")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._lock = Lock()
        self._bucket: dict[tuple[str, int], int] = defaultdict(int)

    @staticmethod
    def _current_minute() -> int:
        return int(time.time() // 60)

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        minute = self._current_minute()
        with self._lock:
            key = (client_ip, minute)
            self._bucket[key] += 1
            current = self._bucket[key]
            stale = [k for k in self._bucket.keys() if k[1] < minute - 2]
            for k in stale:
                self._bucket.pop(k, None)
        remaining = max(self.per_minute - current, 0)
        return current <= self.per_minute, remaining


training_state = TrainingState()
security_config = SecurityConfig()
rate_limiter = RateLimiter(per_minute=security_config.rate_limit_per_minute)

PUBLIC_API_PATHS = {"/api/auth/token"}

gpu_agent: Optional[GPUPurchaseAgent] = None
repository: Optional[RepositoryProtocol] = None


def get_gpu_agent() -> GPUPurchaseAgent:
    global gpu_agent
    if gpu_agent is None:
        gpu_agent = GPUPurchaseAgent()
    return gpu_agent


def get_repository() -> RepositoryProtocol:
    global repository
    if repository is None:
        backend = os.getenv("GPU_ADVISOR_DB_BACKEND", "sqlite").strip().lower()
        db_path_env = os.getenv("GPU_ADVISOR_DB_PATH", "").strip()
        sqlite_path = Path(db_path_env) if db_path_env else (PROJECT_ROOT / "data" / "processed" / "gpu_advisor.db")
        postgres_dsn = os.getenv("GPU_ADVISOR_POSTGRES_DSN", "").strip() or None
        repository = create_repository(backend=backend, sqlite_path=sqlite_path, postgres_dsn=postgres_dsn)
        repository.initialize()
    return repository


def _load_latest_news_snapshot(project_root: Path) -> Optional[dict]:
    news_dir = project_root / "data" / "raw" / "news"
    if not news_dir.exists():
        return None
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return None
    return NewsSentimentAnalyzer().analyze_news_file(files[-1])


def _apply_realtime_news_to_state(state_vec: np.ndarray, project_root: Path) -> tuple[np.ndarray, dict]:
    snapshot = _load_latest_news_snapshot(project_root)
    if snapshot is None:
        return state_vec, {"applied": False, "reason": "news_snapshot_not_found"}

    vec = np.asarray(state_vec, dtype=np.float32).copy()
    news_start = 80  # price60 + exchange20
    if len(vec) < news_start + 4:
        return vec, {"applied": False, "reason": "state_vector_too_short", "news": snapshot}

    vec[news_start + 0] = float((float(snapshot.get("sentiment_avg", 0.0)) + 1.0) / 2.0)
    vec[news_start + 1] = float(min(float(snapshot.get("total", 0)) / 100.0, 1.0))
    vec[news_start + 2] = float(min(float(snapshot.get("positive_count", 0)) / 50.0, 1.0))
    vec[news_start + 3] = float(min(float(snapshot.get("negative_count", 0)) / 50.0, 1.0))

    return vec, {"applied": True, "feature_start_index": news_start, "news": snapshot}


def _data_readiness() -> dict:
    targets = {
        "danawa": PROJECT_ROOT / "data" / "raw" / "danawa",
        "exchange": PROJECT_ROOT / "data" / "raw" / "exchange",
        "news": PROJECT_ROOT / "data" / "raw" / "news",
        "dataset": PROJECT_ROOT / "data" / "processed" / "dataset",
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


def _validate_auth(request: Request) -> tuple[bool, str]:
    mode = security_config.auth_mode
    if mode == AuthMode.NONE:
        return True, "anonymous"

    auth_header = request.headers.get("authorization", "")
    if security_config.allows_jwt() and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token:
            payload = security_config.jwt.verify_token(token)
            return True, str(payload.get("sub", "unknown"))

    if security_config.allows_api_key() and security_config.authenticate_api_key(request.headers.get("x-api-key")):
        return True, "api_key"

    return False, ""


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    started = time.time()
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path

    remaining = rate_limiter.per_minute
    auth_subject = "anonymous"

    if path.startswith("/api/"):
        allowed, remaining = rate_limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(rate_limiter.per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        if path not in PUBLIC_API_PATHS and security_config.requires_auth():
            try:
                ok, auth_subject = _validate_auth(request)
            except Exception as e:
                return JSONResponse(status_code=401, content={"detail": f"Unauthorized: {e}"})
            if not ok:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.per_minute)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    elapsed_ms = (time.time() - started) * 1000.0
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "client_ip": client_ip,
                "auth_subject": auth_subject,
            },
            ensure_ascii=False,
        )
    )
    return response


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.post("/api/auth/token")
async def issue_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not security_config.allows_jwt():
        raise HTTPException(status_code=400, detail="JWT auth mode is disabled")

    if not security_config.authenticate_password(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = security_config.jwt.issue_token(subject=form_data.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": security_config.jwt_expiry_seconds,
        "auth_mode": security_config.auth_mode.value,
    }


@app.get("/api/agent/readiness")
def agent_readiness():
    readiness = _data_readiness()
    status = "production_candidate" if readiness["ready_for_30d_training"] else "collect_more_data"
    return {"status": status, **readiness}


@app.get("/api/agent/model-info")
def agent_model_info():
    agent = get_gpu_agent()
    return agent.get_model_info()


@app.get("/api/agent/evaluate")
def agent_evaluate(lookback_days: int = 30):
    agent = get_gpu_agent()
    evaluator = AgentEvaluator(project_root=PROJECT_ROOT, agent=agent)
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
    pipeline = AgentReleasePipeline(project_root=PROJECT_ROOT)
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

        resolve_state = getattr(agent, "resolve_state", None)
        decide_from_state = getattr(agent, "decide_from_state", None)
        resolved = None
        if callable(resolve_state) and callable(decide_from_state):
            resolved = resolve_state(query.model_name)

        if isinstance(resolved, tuple) and len(resolved) == 3:
            resolved_model, state_vec, data_date = resolved
            enriched_state_vec, news_context = _apply_realtime_news_to_state(state_vec, PROJECT_ROOT)
            decision = decide_from_state(resolved_model, enriched_state_vec, data_date)

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


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    if training_state.is_training:
        raise HTTPException(status_code=400, detail="Training already running")

    training_state.reset()
    training_state.is_training = True
    training_state.total_steps = config.num_steps
    training_state.start_time = time.time()
    training_state.config = config

    async def training_loop():
        global gpu_agent
        try:
            trainer = AgentFineTuner(project_root=PROJECT_ROOT)

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
            training_state.metrics_history.append({"type": "error", "message": str(e), "timestamp": time.time()})
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
    training_state.is_training = False
    return {"status": "stopped", "steps_completed": training_state.current_step}


@app.get("/api/training/status")
async def training_status():
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
    history = training_state.metrics_history
    if last_n > 0:
        history = history[-last_n:]
    return {"metrics": history, "total": len(training_state.metrics_history)}


@app.get("/api/training/metrics/stream")
async def stream_metrics():
    async def event_generator():
        last_index = 0
        while True:
            if last_index < len(training_state.metrics_history):
                new_metrics = training_state.metrics_history[last_index:]
                for metric in new_metrics:
                    yield f"data: {json.dumps(metric)}\\n\\n"
                last_index = len(training_state.metrics_history)

            if not training_state.is_training and last_index >= len(training_state.metrics_history):
                yield f"data: {json.dumps({'type': 'done'})}\\n\\n"
                break

            await asyncio.sleep(0.2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/training/summary")
async def training_summary():
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
        "security": {
            "auth_mode": security_config.auth_mode.value,
            "jwt_enabled": security_config.allows_jwt(),
            "api_key_enabled": security_config.allows_api_key(),
            "rate_limit_per_minute": rate_limiter.per_minute,
        },
        "persistence": get_repository().health(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
