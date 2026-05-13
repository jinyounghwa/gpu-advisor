import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.security import AuthMode, SecurityConfig
from backend.server.state import RateLimiter
from backend.server.routes.agent import router as agent_router
from backend.server.routes.auth import router as auth_router
from backend.server.routes.system import router as system_router
from backend.server.routes.training import router as training_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU Advisor + AI Training Dashboard")
security_config = SecurityConfig()
rate_limiter = RateLimiter(per_minute=security_config.rate_limit_per_minute)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_API_PATHS = {"/api/auth/token"}

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

    remaining = ""
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
    if remaining != "":
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

app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(system_router)
app.include_router(training_router)

if __name__ == "__main__":
    uvicorn.run("backend.server.app:app", host="0.0.0.0", port=8000, reload=False)
