"""Legacy AlphaZero routes shim.

This module is intentionally minimized.
- Primary production API lives in `backend.simple_server`.
- Kept only for backward compatibility with older scripts importing this path.
"""

from __future__ import annotations

from fastapi import APIRouter

try:
    from simple_server import app as app  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from backend.simple_server import app as app  # type: ignore


router = APIRouter(prefix="/api/alphazero", tags=["alphazero-legacy"])


@router.get("/")
def legacy_info():
    return {
        "status": "deprecated",
        "message": "Legacy alpha routes are deprecated. Use /api/ask and /api/agent/* endpoints.",
        "migration": {
            "inference": "/api/ask",
            "model_info": "/api/agent/model-info",
            "training_status": "/api/training/status",
        },
    }


app.include_router(router)
