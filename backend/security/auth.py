"""JWT/OAuth-like auth utilities without external crypto dependencies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from enum import Enum
from typing import Dict


class AuthMode(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    HYBRID = "hybrid"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


class JWTAuthManager:
    def __init__(self, secret_key: str, expiry_seconds: int = 3600):
        if not secret_key.strip():
            raise ValueError("JWT secret key must not be empty")
        self.secret_key = secret_key.encode("utf-8")
        self.expiry_seconds = expiry_seconds

    def _sign(self, signing_input: bytes) -> str:
        digest = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
        return _b64url_encode(digest)

    def issue_token(self, *, subject: str, role: str = "user") -> str:
        now = int(time.time())
        payload = {
            "sub": subject,
            "role": role,
            "iat": now,
            "exp": now + self.expiry_seconds,
        }
        header = {"alg": "HS256", "typ": "JWT"}

        encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        signature = self._sign(signing_input)
        return f"{encoded_header}.{encoded_payload}.{signature}"

    def verify_token(self, token: str) -> Dict[str, str]:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        encoded_header, encoded_payload, signature = parts
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = self._sign(signing_input)
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Invalid token signature")

        payload_raw = _b64url_decode(encoded_payload)
        payload = json.loads(payload_raw.decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if exp < int(time.time()):
            raise ValueError("Token expired")
        return payload


class SecurityConfig:
    def __init__(self) -> None:
        mode_raw = os.getenv("GPU_ADVISOR_AUTH_MODE", "none").strip().lower()
        self.auth_mode = AuthMode(mode_raw) if mode_raw in {m.value for m in AuthMode} else AuthMode.NONE

        keys_raw = os.getenv("GPU_ADVISOR_API_KEYS", "")
        self.api_keys = {k.strip() for k in keys_raw.split(",") if k.strip()}

        users_raw = os.getenv("GPU_ADVISOR_AUTH_USERS", "")
        self.users = self._parse_users(users_raw)
        if not self.users:
            default_user = os.getenv("GPU_ADVISOR_AUTH_DEFAULT_USER", "admin")
            default_password = os.getenv("GPU_ADVISOR_AUTH_DEFAULT_PASSWORD", "change-me")
            self.users[default_user] = default_password

        self.jwt_expiry_seconds = int(os.getenv("GPU_ADVISOR_JWT_EXP_SECONDS", "3600"))
        self.jwt_secret = os.getenv("GPU_ADVISOR_JWT_SECRET", "gpu-advisor-dev-secret")
        self.jwt = JWTAuthManager(secret_key=self.jwt_secret, expiry_seconds=self.jwt_expiry_seconds)

        self.rate_limit_per_minute = int(os.getenv("GPU_ADVISOR_RATE_LIMIT_PER_MINUTE", "60"))

    @staticmethod
    def _parse_users(raw: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for chunk in raw.split(","):
            token = chunk.strip()
            if not token or ":" not in token:
                continue
            user, pw = token.split(":", 1)
            if user.strip():
                result[user.strip()] = pw
        return result

    def authenticate_api_key(self, api_key: str | None) -> bool:
        return bool(api_key and api_key in self.api_keys)

    def authenticate_password(self, username: str, password: str) -> bool:
        return bool(username in self.users and self.users[username] == password)

    def requires_auth(self) -> bool:
        return self.auth_mode != AuthMode.NONE

    def allows_api_key(self) -> bool:
        return self.auth_mode in {AuthMode.API_KEY, AuthMode.HYBRID}

    def allows_jwt(self) -> bool:
        return self.auth_mode in {AuthMode.JWT, AuthMode.HYBRID}
