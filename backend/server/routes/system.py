import os
import time
import psutil
from fastapi import APIRouter

from backend.server.state import training_state
from backend.server.dependencies import get_repository

router = APIRouter(prefix="/api/system", tags=["system"])


def _get_security_config():
    from backend.server.app import security_config
    return security_config

@router.get("/status")
async def system_status():
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    snapshot = training_state.snapshot()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "memory_mb": round(memory.rss / (1024 * 1024), 1),
        "memory_percent": round(psutil.virtual_memory().percent, 1),
        "disk_percent": round(psutil.disk_usage("/").percent, 1),
        "is_training": snapshot["is_training"],
        "uptime": round(time.time() - (snapshot["start_time"] or time.time()), 1),
        "security": {
            "auth_mode": _get_security_config().auth_mode.value,
            "jwt_enabled": _get_security_config().allows_jwt(),
            "api_key_enabled": _get_security_config().allows_api_key(),
        },
        "persistence": get_repository().health(),
    }
