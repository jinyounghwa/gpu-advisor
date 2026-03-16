"""Legacy AlphaZero routes shim.

This module is intentionally minimized.
- Primary production API lives in `backend.simple_server`.
- Kept only for backward compatibility with older scripts importing this path.
- NOTE: This router is NOT auto-registered. Callers must explicitly call
  `app.include_router(alphazero_routes.router)` if needed.
"""

from __future__ import annotations

from fastapi import APIRouter

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
