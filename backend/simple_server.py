"""
Backward-compatible server entry point.

Prefer using ``python -m backend.server.app`` directly.
This module re-exports ``app`` so that existing tooling
(``uvicorn backend.simple_server:app``) continues to work.
"""

import uvicorn
from backend.server.app import app, security_config  # noqa: F401
from backend.security import AuthMode  # noqa: F401
from backend.server.dependencies import get_gpu_agent  # noqa: F401

if __name__ == "__main__":
    uvicorn.run("backend.simple_server:app", host="0.0.0.0", port=8000, reload=False)
