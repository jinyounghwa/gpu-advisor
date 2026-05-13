import os
from pathlib import Path
from threading import Lock
from typing import Optional

from backend.agent import GPUPurchaseAgent
from backend.storage import RepositoryProtocol, create_repository

PROJECT_ROOT = Path(__file__).resolve().parents[2]

gpu_agent: Optional[GPUPurchaseAgent] = None
repository: Optional[RepositoryProtocol] = None
gpu_agent_lock = Lock()
repository_lock = Lock()

def get_gpu_agent() -> GPUPurchaseAgent:
    global gpu_agent
    with gpu_agent_lock:
        if gpu_agent is None:
            gpu_agent = GPUPurchaseAgent(project_root=PROJECT_ROOT)
    return gpu_agent

def set_gpu_agent(agent: Optional[GPUPurchaseAgent]):
    global gpu_agent
    with gpu_agent_lock:
        gpu_agent = agent

def get_repository() -> RepositoryProtocol:
    global repository
    with repository_lock:
        if repository is None:
            backend = os.getenv("GPU_ADVISOR_DB_BACKEND", "sqlite").strip().lower()
            db_path_env = os.getenv("GPU_ADVISOR_DB_PATH", "").strip()
            sqlite_path = Path(db_path_env) if db_path_env else (PROJECT_ROOT / "data" / "processed" / "gpu_advisor.db")
            postgres_dsn = os.getenv("GPU_ADVISOR_POSTGRES_DSN", "").strip() or None
            repository = create_repository(backend=backend, sqlite_path=sqlite_path, postgres_dsn=postgres_dsn)
            repository.initialize()
    return repository
