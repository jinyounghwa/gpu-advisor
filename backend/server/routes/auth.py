from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _get_security_config():
    """Lazy import to share the singleton from app.py."""
    from backend.server.app import security_config
    return security_config


@router.post("/token")
async def issue_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    security_config = _get_security_config()
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
